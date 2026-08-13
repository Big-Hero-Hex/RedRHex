import unittest

from tools.training_panel.training_panel.server import PanelHandler


class PanelHandlerErrorTests(unittest.TestCase):
    def test_unexpected_request_error_returns_json_instead_of_dropping_connection(self):
        handler = object.__new__(PanelHandler)
        responses = []
        handler.command = "GET"
        handler.path = "/api/deploy/defaults"
        handler._json = lambda payload, status=200: responses.append((payload, status))

        handler._handle_request(lambda: (_ for _ in ()).throw(RuntimeError("dependency probe failed")))

        self.assertEqual(responses, [({"error": "Internal server error: dependency probe failed"}, 500)])


if __name__ == "__main__":
    unittest.main()
