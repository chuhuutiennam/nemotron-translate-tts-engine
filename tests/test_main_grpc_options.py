from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

import grpc
import grpc.aio

from nemotron_testbed.config import Settings
from nemotron_testbed.main import grpc_server_options


def _settings(**env_extra) -> Settings:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GRPC_")}
    env.pop("ENGINE_CONFIG_FILE", None)
    env.pop("APP_DIR", None)
    env["MODEL_PATH"] = "models/asr.nemo"
    env.update(env_extra)
    with patch.dict(os.environ, env, clear=True):
        return Settings.from_env()


class GrpcServerOptionsTest(unittest.TestCase):
    def test_default_options_carry_the_configured_ping_policy(self):
        options = dict(grpc_server_options(_settings()))

        self.assertEqual(options["grpc.keepalive_permit_without_calls"], 1)
        self.assertEqual(options["grpc.http2.min_ping_interval_without_data_ms"], 60000)
        self.assertEqual(options["grpc.http2.max_ping_strikes"], 2)
        self.assertEqual(options["grpc.keepalive_time_ms"], 60000)
        self.assertEqual(options["grpc.keepalive_timeout_ms"], 20000)

    def test_message_size_options_are_kept(self):
        options = dict(grpc_server_options(_settings(MAX_RPC_MESSAGE_MB="3")))

        self.assertEqual(options["grpc.max_receive_message_length"], 3 * 1024 * 1024)
        self.assertEqual(options["grpc.max_send_message_length"], 3 * 1024 * 1024)

    def test_settings_values_reach_the_options(self):
        options = dict(
            grpc_server_options(
                _settings(
                    GRPC_KEEPALIVE_TIME_MS="90000",
                    GRPC_KEEPALIVE_TIMEOUT_MS="15000",
                    GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS="false",
                    GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS="10000",
                    GRPC_MAX_PING_STRIKES="5",
                )
            )
        )

        self.assertEqual(options["grpc.keepalive_time_ms"], 90000)
        self.assertEqual(options["grpc.keepalive_timeout_ms"], 15000)
        self.assertEqual(options["grpc.keepalive_permit_without_calls"], 0)
        self.assertEqual(options["grpc.http2.min_ping_interval_without_data_ms"], 10000)
        self.assertEqual(options["grpc.http2.max_ping_strikes"], 5)


class GrpcServerPingPolicyTest(unittest.IsolatedAsyncioTestCase):
    async def _call_outcome(self, server_env: dict, client_ping_ms: int, wait_s: float):
        async def handler(request_iterator, context):
            async for _ in request_iterator:
                yield b"ok"

        class Handlers(grpc.GenericRpcHandler):
            def service(self, details):
                return grpc.stream_stream_rpc_method_handler(handler)

        async def requests():
            yield b"hi"
            await asyncio.sleep(60)

        server = grpc.aio.server(options=grpc_server_options(_settings(**server_env)))
        server.add_generic_rpc_handlers((Handlers(),))
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        self.addAsyncCleanup(server.stop, None)

        client_options = [
            ("grpc.keepalive_time_ms", client_ping_ms),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", client_ping_ms),
        ]
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{port}", options=client_options
        ) as channel:
            call = channel.stream_stream("/t/Rec")(requests())
            async for _ in call:
                break
            try:
                code = await asyncio.wait_for(call.code(), timeout=wait_s)
            except asyncio.TimeoutError:
                call.cancel()
                return None
            return code

    async def test_client_pinging_faster_than_the_limit_loses_the_connection(self):
        code = await self._call_outcome({}, client_ping_ms=500, wait_s=15)

        self.assertEqual(code, grpc.StatusCode.UNAVAILABLE)

    async def test_ping_strikes_zero_keeps_the_connection(self):
        code = await self._call_outcome(
            {"GRPC_MAX_PING_STRIKES": "0"}, client_ping_ms=500, wait_s=6
        )

        self.assertIsNone(code)


if __name__ == "__main__":
    unittest.main()
