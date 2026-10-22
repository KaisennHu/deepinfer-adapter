# SPDX-License-Identifier: Apache-2.0
"""Worker-contract conformance checks, run against the echo backend.

Any adapter that claims to be a ``grpc://`` worker should pass these. They need
no engine, no weights and no accelerator::

    python -m unittest discover -s tests -v

A failure here means the router would either mis-route to this worker or receive
a successful response that silently ignored part of the request.
"""

from __future__ import annotations

import unittest

try:
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    from deepinfer_adapter import server as adapter_server
    from deepinfer_adapter.backends import BackendCapabilities, EchoBackend

    import control_pb2
    import control_pb2_grpc
    import inference_pb2
    import inference_pb2_grpc
except ImportError as exc:  # no grpcio, or proto_gen was not generated
    raise unittest.SkipTest(f"adapter bindings unavailable: {exc}") from exc

MODEL = "echo-model"
MAX_MODEL_LEN = 8192
KV_BLOCK_SIZE = 64


class WorkerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = EchoBackend()
        cls.server, port = adapter_server.build_server(
            cls.backend, MODEL, MAX_MODEL_LEN, KV_BLOCK_SIZE, "127.0.0.1:0"
        )
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        cls.inference = inference_pb2_grpc.InferenceStub(cls.channel)
        cls.control = control_pb2_grpc.ControlStub(cls.channel)
        cls.health = health_pb2_grpc.HealthStub(cls.channel)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.channel.close()
        cls.server.stop(grace=None)
        cls.backend.close()

    def request(self, ids: list[int], **stopping: object) -> "inference_pb2.GenerateRequest":
        return inference_pb2.GenerateRequest(
            request_id="req-1",
            model=MODEL,
            token_ids=inference_pb2.TokenIds(ids=ids),
            stopping=inference_pb2.StoppingCriteria(**stopping),
        )

    def collect(self, ids: list[int], **stopping: object) -> tuple[list[int], str | None]:
        """Run a streaming request and return its output ids and finish reason."""
        token_ids: list[int] = []
        finish = None
        for event in self.inference.GenerateStream(self.request(ids, **stopping)):
            if event.HasField("outputs"):
                token_ids.extend(event.outputs.token_ids)
                if event.outputs.HasField("finish_info"):
                    finish = inference_pb2.FinishInfo.FinishReason.Name(
                        event.outputs.finish_info.finish_reason
                    )
        return token_ids, finish

    def test_health_reports_serving_with_empty_service(self) -> None:
        response = self.health.Check(health_pb2.HealthCheckRequest(service=""))
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)

    def test_server_info_reports_engine_limits(self) -> None:
        info = self.control.GetServerInfo(control_pb2.GetServerInfoRequest())
        self.assertEqual(info.max_model_len, MAX_MODEL_LEN)
        self.assertEqual(info.kv_block_size, KV_BLOCK_SIZE)
        self.assertTrue(info.instance_id)

    def test_model_info_declares_token_id_input(self) -> None:
        info = self.control.GetModelInfo(control_pb2.GetModelInfoRequest())
        self.assertEqual(info.served_model_name, MODEL)
        self.assertTrue(info.supports_token_ids_input)
        self.assertFalse(info.supports_text_input)

    def test_kv_event_sources_are_empty_rather_than_invented(self) -> None:
        response = self.control.GetKvEventSources(control_pb2.GetKvEventSourcesRequest())
        self.assertEqual(len(response.sources), 0)

    def test_generate_returns_prompt_info_and_output_ids(self) -> None:
        response = self.inference.Generate(self.request([1, 2, 3]))
        self.assertEqual(response.prompt_info.num_prompt_tokens, 3)
        # Echo reverses the prompt, so exact ids prove nothing was re-tokenised.
        self.assertEqual(list(response.outputs.token_ids), [3, 2, 1])
        self.assertEqual(response.outputs.num_tokens, 3)

    def test_generate_stream_sends_prompt_info_before_any_output(self) -> None:
        events = list(self.inference.GenerateStream(self.request([7, 8])))
        self.assertTrue(events[0].HasField("prompt_info"))
        self.assertEqual(events[0].prompt_info.num_prompt_tokens, 2)
        self.assertTrue(all(event.HasField("outputs") for event in events[1:]))

    def test_finish_reason_is_length_at_max_tokens(self) -> None:
        _, finish = self.collect([1, 2], max_new_tokens=2)
        self.assertEqual(finish, "LENGTH")

    def test_stop_token_ids_end_generation_early(self) -> None:
        # Echo emits 4, 3, 2, 1; stopping on 3 must cut the stream after two ids.
        token_ids, finish = self.collect([1, 2, 3, 4], max_new_tokens=4, stop_token_ids=[3])
        self.assertEqual(token_ids, [4, 3])
        self.assertEqual(finish, "STOP")

    def test_stop_strings_are_rejected_not_dropped(self) -> None:
        with self.assertRaises(grpc.RpcError) as raised:
            self.collect([1, 2], stop_strings=["END"])
        self.assertEqual(raised.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("stop_strings", raised.exception.details())

    def test_text_prompt_is_rejected(self) -> None:
        request = inference_pb2.GenerateRequest(request_id="req-2", model=MODEL, text="hello")
        with self.assertRaises(grpc.RpcError) as raised:
            self.inference.Generate(request)
        self.assertEqual(raised.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    def test_empty_prompt_is_rejected(self) -> None:
        with self.assertRaises(grpc.RpcError) as raised:
            self.inference.Generate(self.request([]))
        self.assertEqual(raised.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)


class UnsupportedStopTest(unittest.TestCase):
    """The DeepInfer backend declares no stop support; nothing may be dropped."""

    class NoStopBackend(EchoBackend):
        capabilities = BackendCapabilities()

    def setUp(self) -> None:
        self.servicer = adapter_server.InferenceServicer(self.NoStopBackend(), MODEL)

    def test_stop_token_ids_is_rejected_before_any_output(self) -> None:
        request = inference_pb2.GenerateRequest(
            request_id="req-3",
            model=MODEL,
            token_ids=inference_pb2.TokenIds(ids=[1, 2]),
            stopping=inference_pb2.StoppingCriteria(stop_token_ids=[2]),
        )
        with self.assertRaises(ValueError) as raised:
            next(self.servicer._events(request))
        self.assertIn("stop_token_ids", str(raised.exception))

    def test_no_stop_constraint_is_accepted(self) -> None:
        request = inference_pb2.GenerateRequest(
            request_id="req-4",
            model=MODEL,
            token_ids=inference_pb2.TokenIds(ids=[1, 2]),
            stopping=inference_pb2.StoppingCriteria(max_new_tokens=1),
        )
        first = next(self.servicer._events(request))
        self.assertEqual(first.prompt_info.num_prompt_tokens, 2)


if __name__ == "__main__":
    unittest.main()
