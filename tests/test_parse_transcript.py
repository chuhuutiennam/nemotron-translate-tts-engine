import unittest

from nemotron_testbed.runtime import NemotronRuntime

parse = NemotronRuntime._parse_transcript


class ParseTranscriptTest(unittest.TestCase):
    # --- explicit language ---

    def test_trailing_tag_stripped(self):
        text, lang = parse("xin chào <vi-VN>", "vi")
        self.assertEqual(text, "xin chào")
        self.assertEqual(lang, "vi")

    def test_mid_sentence_tag_stripped(self):
        # regression: model emits tag at sentence boundary, not only at end
        text, lang = parse(
            "Hệ thống đang xử lý khoảng một nghìn hai trăm câu mỗi giây xin chào. <vi-VN> hôm nay",
            "vi",
        )
        self.assertEqual(
            text,
            "Hệ thống đang xử lý khoảng một nghìn hai trăm câu mỗi giây xin chào. hôm nay",
        )
        self.assertEqual(lang, "vi")

    def test_multiple_mid_tags_all_stripped(self):
        text, lang = parse("hello <en-US> world <en-US> bye", "en")
        self.assertEqual(text, "hello world bye")
        self.assertEqual(lang, "en")

    def test_case_insensitive(self):
        text, lang = parse("xin chào <VI-VN>", "vi")
        self.assertEqual(text, "xin chào")
        self.assertEqual(lang, "vi")

    def test_no_tag(self):
        text, lang = parse("hello world", "en")
        self.assertEqual(text, "hello world")
        self.assertEqual(lang, "en")

    def test_empty(self):
        text, lang = parse("", "vi")
        self.assertEqual(text, "")
        self.assertEqual(lang, "vi")

    # --- auto language ---

    def test_auto_trailing_tag(self):
        text, lang = parse("xin chào <vi-VN>", "auto")
        self.assertEqual(text, "xin chào")
        self.assertEqual(lang, "vi-VN")

    def test_auto_mid_sentence_tag(self):
        text, lang = parse("xin chào. <vi-VN> hôm nay", "auto")
        self.assertEqual(text, "xin chào. hôm nay")
        self.assertEqual(lang, "vi-VN")

    def test_auto_no_tag(self):
        text, lang = parse("hello world", "auto")
        self.assertEqual(text, "hello world")
        self.assertEqual(lang, "")

    def test_auto_multiple_tags_last_wins(self):
        # code-switch: take last tag as detected language
        text, lang = parse("hello <en-US> xin chào <vi-VN>", "auto")
        self.assertEqual(text, "hello xin chào")
        self.assertEqual(lang, "vi-VN")


if __name__ == "__main__":
    unittest.main()
