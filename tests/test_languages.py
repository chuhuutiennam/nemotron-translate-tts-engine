import unittest

from nemotron_testbed.languages import envit5_prefix, is_translatable_language, normalize_language, short_language


class LanguageTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_language("vi"), "vi-VN")
        self.assertEqual(normalize_language("en"), "en-US")

    def test_envit5_prefix(self):
        self.assertEqual(envit5_prefix("vi-VN", "vi: ", "en: "), "vi: ")

    def test_translatable_languages_are_vi_and_en(self):
        self.assertTrue(is_translatable_language("vi-VN"))
        self.assertTrue(is_translatable_language("en"))
        self.assertFalse(is_translatable_language("pt-BR"))
        self.assertFalse(is_translatable_language(""))

    def test_short_language_compares_regional_variants(self):
        self.assertEqual(short_language("en-US"), short_language("en"))


if __name__ == "__main__":
    unittest.main()
