from ser.word2vec_wrapper import Word2VecWrapper


class Preprocessor:
    sentence_len = []

    @staticmethod
    def preprocess_one(raw_text):
        if (raw_text.startswith("\'") and raw_text.endswith("\'")) or (raw_text.startswith("\"") and raw_text.endswith("\"")):
            raw_text = raw_text[1:-1]
        text = raw_text.lower()
        text = filter(text, Filters.is_invalid)
        text = text.replace(".", " . ").replace(",", " , ")
        text = text.replace("!", " ! ").replace("?", " ? ")
        text = text.replace("%", " percent ")
        text = filter(text, Filters._is_empty)
        text = filter(text, Filters._not_in_vocab)

        return text

    @staticmethod
    def preprocess_many(text_list):
        return [Preprocessor.preprocess_one(text) for text in text_list if len(text.split(" ")) < 30]


def filter(text, filter_func):
    words = text.split(" ")
    valid_words = [word for word in words if not filter_func(word)]
    return " ".join(valid_words)


class Filters:
    @staticmethod
    def _is_empty(word):
        return word == "" or word == " " or word == "\t" or word == "\n";

    @staticmethod
    def _is_number(word):
        return word.isdigit()

    @staticmethod
    def _is_special(word):
        return word == "-" or word == "-" or word == "/" or word == "(" or word == ")"

    @staticmethod
    def _not_in_vocab(word):
        return not Word2VecWrapper.vocab_contains(word)

    @staticmethod
    def is_invalid(word):
        for filter in Filters.all():
            if filter(word):
                return True
        return False

    @staticmethod
    def all():
        return [Filters._is_number, Filters._is_empty, Filters._is_special]

