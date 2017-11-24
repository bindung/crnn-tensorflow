label_to_char_dict = None
char_to_label_dict = None


def build_label():
    global label_to_char_dict
    global char_to_label_dict

    label_to_char_dict = {}
    char_to_label_dict = {}

    def append_dict(c, idx):
        char_to_label_dict[c] = idx
        label_to_char_dict[idx] = c
        return idx + 1

    idx = append_dict(' ', 0)

    for i in range(10):
        c = chr(ord('0') + i)
        idx = append_dict(c, idx)

    for i in range(26):
        c = chr(ord('a') + i)
        idx = append_dict(c, idx)

    for i in range(26):
        c = chr(ord('A') + i)
        idx = append_dict(c, idx)

    idx = append_dict('@', idx)
    idx = append_dict('-', idx)
    idx = append_dict('\'', idx)
    idx = append_dict('"', idx)
    idx = append_dict('!', idx)
    idx = append_dict('.', idx)
    idx = append_dict('{', idx)
    idx = append_dict('}', idx)
    idx = append_dict('(', idx)
    idx = append_dict(')', idx)
    idx = append_dict('/', idx)
    idx = append_dict('\\', idx)
    idx = append_dict(':', idx)
    idx = append_dict('&', idx)
    idx = append_dict('?', idx)
    idx = append_dict('>', idx)
    idx = append_dict('<', idx)


def char_to_label(c):
    return ord(c)


def dict_to_label(l):
    return chr(l)
