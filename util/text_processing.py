"""
Writer : Team  Murphy-s-rule (Team Nandoc)

This module contain functions or methods to process text for label in datasets or output of model.
To generate HTML documentation for this module issue the command:

    pydoc -w util/text_processing.py

"""

from enum import Enum
import re
from jamo import h2j, j2hcj
from util.unicode import join_jamos
from tensorflow_text.python.ops.tokenization import Detokenizer
from tensorflow_text.python.ops.tokenization import Tokenizer

class HangulTokenType(Enum):
    """
    tokentype
        - onset:  ㄱ | ㄲ | ㄴ | ㄷ | ㄸ | ㄹ | ㅁ | ㅂ | ㅃ | ㅅ | ㅆ | ㅇ | ㅈ | ㅉ | ㅊ | ㅋ | ㅌ | ㅍ | ㅎ
        - nucleus: ㅏ | ㅐ | ㅑ | ㅒ | ㅓ | ㅔ | ㅕ | ㅖ | ㅗ | ㅘ | ㅙ | ㅚ | ㅛ | ㅜ | ㅝ | ㅞ | ㅟ | ㅠ | ㅡ | ㅢ | ㅣ
        - coda: ㄱ | ㄲ | ㄳ | ㄴ | ㄵ | ㄶ | ㄷ | ㄹ | ㄺ | ㄻ | ㄼ | ㄽ | ㄾ | ㄿ | ㅀ | ㅁ | ㅂ | ㅄ | ㅅ | ㅆ | ㅇ | ㅈ | ㅊ | ㅋ | ㅌ | ㅍ | ㅎ
    """
    ONSET = 0
    NUCLEUS = 1
    CODA = 2

map_token_to_num = {
    '<pad>': 0, '<sos>': 1, '<eos>': 2,
    HangulTokenType.ONSET: {
        'ㄱ':  3, 'ㄲ':  4, 'ㄴ':  5, 'ㄷ':  6, 'ㄸ':  7, 'ㄹ':  8, 'ㅁ':  9, 'ㅂ': 10, 'ㅃ': 11, 'ㅅ': 12, 'ㅆ': 13,
        'ㅇ': 14, 'ㅈ': 15, 'ㅉ': 16, 'ㅊ': 17, 'ㅋ': 18, 'ㅌ': 19, 'ㅍ': 20, 'ㅎ': 21},
    HangulTokenType.NUCLEUS: {
        'ㅏ': 22, 'ㅐ': 23, 'ㅑ': 24, 'ㅒ': 25, 'ㅓ': 26, 'ㅔ': 27, 'ㅕ': 28, 'ㅖ': 29, 'ㅗ': 30, 'ㅘ': 31, 'ㅙ': 32,
        'ㅚ': 33, 'ㅛ': 34, 'ㅜ': 35, 'ㅝ': 36, 'ㅞ': 37, 'ㅟ': 38, 'ㅠ': 39, 'ㅡ': 40, 'ㅢ': 41, 'ㅣ': 42},
    HangulTokenType.CODA: {
        'ㅏ': 22, 'ㅐ': 23, 'ㅑ': 24, 'ㅒ': 25, 'ㅓ': 26, 'ㅔ': 27, 'ㅕ': 28, 'ㅖ': 29, 'ㅗ': 30, 'ㅘ': 31, 'ㅙ': 32,
        'ㅚ': 33, 'ㅛ': 34, 'ㅜ': 35, 'ㅝ': 36, 'ㅞ': 37, 'ㅟ': 38, 'ㅠ': 39, 'ㅡ': 40, 'ㅢ': 41, 'ㅣ': 42,
        'ㄱ': 43, 'ㄲ': 44, 'ㄳ': 45, 'ㄴ': 46, 'ㄵ': 47, 'ㄶ': 48, 'ㄷ': 49, 'ㄸ': 50, 'ㄹ': 51, 'ㄺ': 52, 'ㄻ': 53, 'ㄼ': 54,
        'ㄽ': 55, 'ㄾ': 56, 'ㄿ': 57, 'ㅀ': 58, 'ㅁ': 59, 'ㅂ': 60, 'ㅃ': 61, 'ㅄ': 62, 'ㅅ': 63, 'ㅆ': 64, 'ㅇ': 65, 'ㅈ': 66,
        'ㅉ': 67, 'ㅊ':  68, 'ㅋ': 69, 'ㅌ': 70, 'ㅍ': 71, 'ㅎ': 72},
    'a': 73, 'b': 74, 'c': 75, 'd': 76, 'e': 77, 'f': 78, 'g': 79, 'h': 80, 'i': 81, 'j': 82, 'k': 83, 'l': 84, 'm': 85,
    'n': 86, 'o': 87, 'p': 88, 'q': 89, 'r': 90, 's': 91, 't': 92,
    'u': 93, 'v': 94, 'w': 95, 'x': 96, 'y': 97, 'z': 98,
    'A':  99, 'B':  100, 'C':  101, 'D':  102, 'E': 103, 'F': 104, 'G': 105, 'H': 106, 'I': 107, 'J': 108,
    'K': 109, 'L': 110, 'M': 111, 'N': 112, 'O': 113, 'P': 114, 'Q': 115, 'R': 116, 'S': 117, 'T': 118,
    'U': 119, 'V': 120, 'W': 121, 'X': 122, 'Y': 123, 'Z': 124,
    '0': 125, '1': 126, '2': 127, '3': 128, '4': 129, '5': 130, '6': 131, '7': 132, '8': 133, '9': 134,
    '!': 135, '@': 136, '#': 137, '$': 138, '%': 139, '^': 140, '&': 141, '*': 142, '(': 143, ')': 144, '-': 145,
    '_': 146, '=': 147, '+': 148, '[': 149, ']': 150, '{': 151, '}': 152, '\\': 153, '|': 154, ':': 155, ';': 156,
    "'": 157, '"': 158, ',': 159, '.': 160, '<': 161, '>': 162, '/': 163, '?': 164, ' ': 165, '`': 166, '~': 167,
    '•': 168, '<unk>': 169,
}

map_num_to_token = [
    '<pad>', '<sos>', '<eos>',
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
    'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '[', ']', '{', '}', '\\', '|', ':', ';', "'",
    '"', ',', '.', '<', '>', '/', '?', ' ', '`', '~', '•', '<unk>'
]

def isHangul(token):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', token))
    return hanCount > 0


def token_to_num(token):
    """
    The token_to_num function returns the number mapped by input token

    Args:
        token (str): The token will be converted to number mapped

    Returns:
        list(int): The numbers converted from input token
    """
    if isHangul(token):
        subtokens = j2hcj(h2j(token))
        return [map_token_to_num[HangulTokenType(i)][sub_token] for i, sub_token in enumerate(subtokens)] \
                if len(subtokens) > 1 else [map_token_to_num[HangulTokenType.CODA][token]]
    else:
        return [map_token_to_num[token]]

def num_to_token(nums):
    """
        The nums_to_token returns the token mapped by input numbers

        Args:
            list(int): The numbers will be converted to token mapped

        Returns:
            token (str): The token converted from input numbers
    """
    return join_jamos([map_num_to_token[num] for num in nums]) if len(nums) > 1 else map_num_to_token[nums[0]]

def tokenize_text(text):
    """
        The tokenize_text function returns the tokens from text, always start token is <sos>, end token is <eos>

        Args:
            text (str): The text will be tokenized

        Returns:
            list(str): The tokens from input text
    """
    return ['<sos>'] + [token if token in map_num_to_token or isHangul(token) else '<unk>' for token in list(text)] + ['<eos>']

def split_nums(nums):
    """
        The split_nums function returns the list split numbers by each token

        Args:
            nums (list(int)): The number sequence will be splited by each token

        Returns:
            list(list(int)): The splited numbers from input full number sequence
    """
    nums_splited = []
    for i, num in enumerate(nums):
        if len(nums_splited) > 0 and ((num in range(22,43) and nums[i-1] in range(3,22)) or \
           (len(nums_splited[-1]) == 2 and num in range(43,73) and nums[i-1] in range(22,43) and nums[i-2] in range(3,22))):
            nums_splited[-1].append(num)
            continue
        nums_splited.append([num])

    return nums_splited

def text_to_nums(text):
    """
        The text_to_nums function returns the number list converted from text.
        The input text is tokenized by function, and after that tokens is converted to number list.

        Args:
            text (str): The text will be converted to numbers

        Returns:
            list(int): The numbers converted from tokens of text
    """
    text = text.replace('‚',',')
    text = text.replace('․','.')
    nums = []
    for token in tokenize_text(text):
        nums.extend(token_to_num(token))

    return nums

def nums_to_text(nums):
    """
        The nums_to_text function returns the text converted from number sequence.
        The input number sequence is splited by function, and after that numbers splited is converted to token.

        Args:
            nums (list(int)): The number sequence will be converted to text

        Returns:
            str: The text converted from number sequence
    """
    return ''.join([num_to_token(num_splited) for num_splited in split_nums(nums)][1:-1])

class HangulTokenizer(Tokenizer, Detokenizer):
    char_size = len(map_num_to_token)
    def tokenize(self, input, max_length=None):
        nums = text_to_nums(input)
        if max_length:
            margin_length = max_length - len(nums)
            nums = nums + (token_to_num('<pad>') * margin_length) if max_length > 0 else nums[:margin_length]
        return nums
    def detokenize(self, input):
        nums = input.to_list()
        return nums_to_text(nums)

