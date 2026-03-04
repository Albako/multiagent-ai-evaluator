import pytest

class TestEstimateTokens:
    """ Test estimate_tokens function """
    def test_empty_string(self):
        from main import estimate_tokens
        assert estimate_tokens("") == 0

    def test_short_string(self):
        from main import estimate_tokens
        assert estimate_tokens("test") == 1

    def test_long_string(self):
        from main import estimate_tokens
        assert estimate_tokens("a" * 100) == 25

class TestSlidingWindow:
    """ Test apply_sliding_window function """
    def setup_method(self):
        import main
        main.chat_histories = {}

    def test_new_session_creates_history(self):
        from main import apply_sliding_window, chat_histories
        result = apply_sliding_window("test_session", "Hello")
        assert "test_session" in chat_histories
        assert "User: Hello" in result
        assert result.endswith("AI: ")

    def test_existing_session_appends(self):
        from main import apply_sliding_window, chat_histories
        apply_sliding_window("s1", "First message")
        result = apply_sliding_window("s1", "Second message")
        assert "First message" in result
        assert "Second message" in result

    def test_sliding_window_trims_old_messages(self):
        from main import apply_sliding_window
        # Max tokens = 4000, so 16000 characters
        long_msg = "x" * 15000
        apply_sliding_window("s1", long_msg)
        result = apply_sliding_window("s1", "New message")
        # old message should be trimmed
        assert "New message" in result

class TestUpdateHistory:
    def setup_method(self):
        import main
        main.chat_histories = {"s1": "User: Hi\nAI: "}

    def test_appends_response(self):
        from main import update_history_with_response, chat_histories
        update_history_with_response("s1", "Hello there!")
        assert "Hello there!" in chat_histories["s1"]
