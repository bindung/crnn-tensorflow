def char_to_label(c):
    if c == ' ':
        return 0
    idx = ord(c) - ord('A')
    if idx < 0 or idx >= 26:
        raise ValueError("invalid char")
    return idx + 1


def label_to_char(l):
    if l == 0:
        return ' '
    if l > 26:
        raise ValueError("invalid char")
    return chr(l + ord('A'))
