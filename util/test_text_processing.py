import unittest
import functools
from util import text_processing

test_sheet_is_hangul = [
    {'args': '파', 'return': True}, {'args': '이', 'return': True}, {'args': '썬', 'return': True}, {'args': '에', 'return': True},
    {'args': '서', 'return': True}, {'args': '한', 'return': True}, {'args': '글', 'return': True}, {'args': '인', 'return': True},
    {'args': '지', 'return': True}, {'args': '영', 'return': True}, {'args': '어', 'return': True}, {'args': '인', 'return': True},
    {'args': '지', 'return': True}, {'args': '구', 'return': True}, {'args': '분', 'return': True}, {'args': '한', 'return': True},
    {'args': '다', 'return': True}, {'args': '여', 'return': True}, {'args': '러', 'return': True}, {'args': '가', 'return': True},
    {'args': '지', 'return': True}, {'args': '방', 'return': True}, {'args': '법', 'return': True}, {'args': '을', 'return': True},
    {'args': '테', 'return': True}, {'args': '스', 'return': True}, {'args': '트', 'return': True}, {'args': '함', 'return': True},
    {'args': 'a', 'return': False}, {'args': 'b', 'return': False}, {'args': 'c', 'return': False}, {'args': 'd', 'return': False},
    {'args': 'e', 'return': False}, {'args': 'f', 'return': False}, {'args': 'g', 'return': False}, {'args': 'h', 'return': False},
    {'args': 'i', 'return': False}, {'args': 'j', 'return': False}, {'args': 'k', 'return': False}, {'args': 'l', 'return': False},
    {'args': 'm', 'return': False}, {'args': 'n', 'return': False}, {'args': 'o', 'return': False}, {'args': 'p', 'return': False},
    {'args': 'q', 'return': False}, {'args': 'r', 'return': False}, {'args': 's', 'return': False}, {'args': 't', 'return': False},
    {'args': 'u', 'return': False}, {'args': 'v', 'return': False}, {'args': 'w', 'return': False}, {'args': 'x', 'return': False},
    {'args': 'y', 'return': False}, {'args': 'z', 'return': False}, {'args': 'A', 'return': False}, {'args': 'B', 'return': False},
    {'args': 'C', 'return': False}, {'args': 'D', 'return': False}, {'args': 'E', 'return': False}, {'args': 'F', 'return': False},
    {'args': 'G', 'return': False}, {'args': 'H', 'return': False}, {'args': 'I', 'return': False}, {'args': 'J', 'return': False},
    {'args': 'K', 'return': False}, {'args': 'L', 'return': False}, {'args': 'M', 'return': False}, {'args': 'N', 'return': False},
    {'args': 'O', 'return': False}, {'args': 'P', 'return': False}, {'args': 'Q', 'return': False}, {'args': 'R', 'return': False},
    {'args': 'S', 'return': False}, {'args': 'T', 'return': False}, {'args': 'U', 'return': False}, {'args': 'V', 'return': False},
    {'args': 'W', 'return': False}, {'args': 'X', 'return': False}, {'args': 'Y', 'return': False}, {'args': 'Z', 'return': False},
    {'args': '!', 'return': False}, {'args': '@', 'return': False}, {'args': '#', 'return': False}, {'args': '$', 'return': False},
    {'args': '%', 'return': False}, {'args': '^', 'return': False}, {'args': '&', 'return': False}, {'args': '*', 'return': False},
    {'args': '(', 'return': False}, {'args': ')', 'return': False}, {'args': '-', 'return': False}, {'args': '_', 'return': False},
    {'args': '=', 'return': False}, {'args': '+', 'return': False}, {'args': '[', 'return': False}, {'args': ']', 'return': False},
    {'args': '{', 'return': False}, {'args': '}', 'return': False}, {'args': '\\', 'return': False}, {'args': '|', 'return': False},
    {'args': ':', 'return': False}, {'args': ';', 'return': False}, {'args': "'", 'return': False}, {'args': '"', 'return': False},
    {'args': ',', 'return': False}, {'args': '.', 'return': False}, {'args': '<', 'return': False}, {'args': '>', 'return': False},
    {'args': '/', 'return': False}, {'args': '?', 'return': False}
]

test_sheet_token_to_num = [
    {'args': '<sos>', 'return': [  0]}, {'args': '<pad>', 'return': [  1]}, {'args': '<eos>', 'return': [  2]},
    {'args': 'ㄱ',    'return': [ 43]}, {'args': 'ㄲ',     'return': [ 44]}, {'args': 'ㄳ',    'return': [ 45]},
    {'args': 'ㄴ',    'return': [ 46]}, {'args': 'ㄵ',     'return': [ 47]}, {'args': 'ㄶ',    'return': [ 48]},
    {'args': 'ㄷ',    'return': [ 49]}, {'args': 'ㄸ',     'return': [ 50]}, {'args': 'ㄹ',    'return': [ 51]},
    {'args': 'ㄺ',    'return': [ 52]}, {'args': 'ㄻ',     'return': [ 53]}, {'args': 'ㄼ',    'return': [ 54]},
    {'args': 'ㄽ',    'return': [ 55]}, {'args': 'ㄾ',     'return': [ 56]}, {'args': 'ㄿ',    'return': [ 57]},
    {'args': 'ㅀ',    'return': [ 58]}, {'args': 'ㅁ',     'return': [ 59]}, {'args': 'ㅂ',    'return': [ 60]},
    {'args': 'ㅃ',    'return': [ 61]}, {'args': 'ㅄ',     'return': [ 62]}, {'args': 'ㅅ',    'return': [ 63]},
    {'args': 'ㅆ',    'return': [ 64]}, {'args': 'ㅇ',     'return': [ 65]}, {'args': 'ㅈ',    'return': [ 66]},
    {'args': 'ㅉ',    'return': [ 67]}, {'args': 'ㅊ',     'return': [ 68]}, {'args': 'ㅋ',    'return': [ 69]},
    {'args': 'ㅌ',    'return': [ 70]}, {'args': 'ㅍ',     'return': [ 71]}, {'args': 'ㅎ',    'return': [ 72]},
    {'args': 'ㅏ',    'return': [ 22]}, {'args': 'ㅐ',     'return': [ 23]}, {'args': 'ㅑ',    'return': [ 24]},
    {'args': 'ㅒ',    'return': [ 25]}, {'args': 'ㅓ',     'return': [ 26]}, {'args': 'ㅔ',    'return': [ 27]},
    {'args': 'ㅕ',    'return': [ 28]}, {'args': 'ㅖ',     'return': [ 29]}, {'args': 'ㅗ',    'return': [ 30]},
    {'args': 'ㅘ',    'return': [ 31]}, {'args': 'ㅙ',     'return': [ 32]}, {'args': 'ㅚ',    'return': [ 33]},
    {'args': 'ㅛ',    'return': [ 34]}, {'args': 'ㅜ',     'return': [ 35]}, {'args': 'ㅝ',    'return': [ 36]},
    {'args': 'ㅞ',    'return': [ 37]}, {'args': 'ㅟ',     'return': [ 38]}, {'args': 'ㅠ',    'return': [ 39]},
    {'args': 'ㅡ',    'return': [ 40]}, {'args': 'ㅢ',     'return': [ 41]}, {'args': 'ㅣ',    'return': [ 42]},
    {'args': 'a',     'return': [ 73]}, {'args': 'b',     'return': [ 74]}, {'args': 'c',     'return': [ 75]},
    {'args': 'd',     'return': [ 76]}, {'args': 'e',     'return': [ 77]}, {'args': 'f',     'return': [ 78]},
    {'args': 'g',     'return': [ 79]}, {'args': 'h',     'return': [ 80]}, {'args': 'i',     'return': [ 81]},
    {'args': 'j',     'return': [ 82]}, {'args': 'k',     'return': [ 83]}, {'args': 'l',     'return': [ 84]},
    {'args': 'm',     'return': [ 85]}, {'args': 'n',     'return': [ 86]}, {'args': 'o',     'return': [ 87]},
    {'args': 'p',     'return': [ 88]}, {'args': 'q',     'return': [ 89]}, {'args': 'r',     'return': [ 90]},
    {'args': 's',     'return': [ 91]}, {'args': 't',     'return': [ 92]}, {'args': 'u',     'return': [ 93]},
    {'args': 'v',     'return': [ 94]}, {'args': 'w',     'return': [ 95]}, {'args': 'x',     'return': [ 96]},
    {'args': 'y',     'return': [ 97]}, {'args': 'z',     'return': [ 98]},
    {'args': 'A',     'return': [ 99]}, {'args': 'B',     'return': [100]}, {'args': 'C',     'return': [101]},
    {'args': 'D',     'return': [102]}, {'args': 'E',     'return': [103]}, {'args': 'F',     'return': [104]},
    {'args': 'G',     'return': [105]}, {'args': 'H',     'return': [106]}, {'args': 'I',     'return': [107]},
    {'args': 'J',     'return': [108]}, {'args': 'K',     'return': [109]}, {'args': 'L',     'return': [110]},
    {'args': 'M',     'return': [111]}, {'args': 'N',     'return': [112]}, {'args': 'O',     'return': [113]},
    {'args': 'P',     'return': [114]}, {'args': 'Q',     'return': [115]}, {'args': 'R',     'return': [116]},
    {'args': 'S',     'return': [117]}, {'args': 'T',     'return': [118]}, {'args': 'U',     'return': [119]},
    {'args': 'V',     'return': [120]}, {'args': 'W',     'return': [121]}, {'args': 'X',     'return': [122]},
    {'args': 'Y',     'return': [123]}, {'args': 'Z',     'return': [124]},
    {'args': '0',     'return': [125]}, {'args': '1',     'return': [126]}, {'args': '2',     'return': [127]},
    {'args': '3',     'return': [128]}, {'args': '4',     'return': [129]}, {'args': '5',     'return': [130]},
    {'args': '6',     'return': [131]}, {'args': '7',     'return': [132]}, {'args': '8',     'return': [133]},
    {'args': '9',     'return': [134]},
    {'args': '!',     'return': [135]}, {'args': '@',     'return': [136]}, {'args': '#',     'return': [137]},
    {'args': '$',     'return': [138]}, {'args': '%',     'return': [139]}, {'args': '^',     'return': [140]},
    {'args': '&',     'return': [141]}, {'args': '*',     'return': [142]}, {'args': '(',     'return': [143]},
    {'args': ')',     'return': [144]}, {'args': '-',     'return': [145]}, {'args': '_',     'return': [146]},
    {'args': '=',     'return': [147]}, {'args': '+',     'return': [148]}, {'args': '[',     'return': [149]},
    {'args': ']',     'return': [150]}, {'args': '{',     'return': [151]}, {'args': '}',     'return': [152]},
    {'args': '\\',    'return': [153]}, {'args': '|',     'return': [154]}, {'args': ':',     'return': [155]},
    {'args': ';',     'return': [156]}, {'args': "'",     'return': [157]}, {'args': '"',     'return': [158]},
    {'args': ',',     'return': [159]}, {'args': '.',     'return': [160]}, {'args': '<',     'return': [161]},
    {'args': '>',     'return': [162]}, {'args': '/',     'return': [163]}, {'args': '?',     'return': [164]},
    {'args': ' ',     'return': [165]},
    {'args': '파', 'return': [20, 22]},     {'args': '이', 'return': [14, 42]},     {'args': '썬', 'return': [13, 26, 46]},
    {'args': '에', 'return': [14, 27]},     {'args': '서', 'return': [12, 26]},     {'args': '한', 'return': [21, 22, 46]},
    {'args': '글', 'return': [3, 40, 51]},  {'args': '인', 'return': [14, 42, 46]}, {'args': '지', 'return': [15, 42]},
    {'args': '영', 'return': [14, 28, 65]}, {'args': '어', 'return': [14, 26]},     {'args': '잘', 'return': [15, 22, 51]},
    {'args': '엣', 'return': [14, 27, 63]}, {'args': '구', 'return': [3, 35]},      {'args': '분', 'return': [10, 35, 46]},
    {'args': '명', 'return': [9, 28, 65]},  {'args': '다', 'return': [6, 22]},      {'args': '여', 'return': [14, 28]},
    {'args': '러', 'return': [8, 26]},      {'args': '가', 'return': [3, 22]},      {'args': '왈', 'return': [14, 31, 51]},
    {'args': '방', 'return': [10, 22, 65]}, {'args': '법', 'return': [10, 26, 60]}, {'args': '을', 'return': [14, 40, 51]},
    {'args': '테', 'return': [19, 27]},     {'args': '스', 'return': [12, 40]},     {'args': '트', 'return': [19, 40]},
    {'args': '함', 'return': [21, 22, 59]},
]

test_sheet_tokenize_text = [
    {
        'args': '이제 seq2seq를 이용해서 기계 번역기를 만들어보도록 하겠습니다.',
        'return': ['<sos>','이','제',' ','s','e','q','2','s','e','q','를',' ','이','용','해','서',' ','기','계',' ','번','역','기',
                   '를',' ','만','들','어','보','도','록',' ','하','겠','습','니','다','.','<eos>']
    },
    {
        'args': '병렬 데이터라고 하면 앞서 수행한 태깅 작업의 데이터를 생각할 수 있지만,',
        'return': ['<sos>','병','렬',' ','데','이','터','라','고',' ','하','면',' ','앞','서',' ','수','행','한',' ','태',
                   '깅',' ','작','업','의',' ','데','이','터','를',' ','생','각','할',' ','수',' ','있','지','만',',','<eos>']
    },
    {
        'args': 'tar는 target의 줄임말로 번역하고자 하는 문장을 나타냅니다.',
        'return': ['<sos>','t','a','r','는',' ','t','a','r','g','e','t','의',' ','줄','임','말','로',' ','번','역',
                   '하','고','자',' ','하','는',' ','문','장','을',' ','나','타','냅','니','다','.','<eos>']
    },
]
test_sheet_text_to_nums = [
    {
        'args': '이제 seq2seq를 이용해서 기계 번역기를 만들어보도록 하겠습니다.',
        'return': [0, 14,42, 15,27, 165, 91, 77, 89, 127, 91, 77, 89, 8,40,51, 165, 14,42, 14,34,65, 21,23, 12,26, 165, 3,42, 3,29, 165,
                   10,26,46, 14,28,43, 3,42, 8,40,51, 165, 9,22,46, 6,40,51, 14,26, 10,30, 6,30, 8,30,43, 165, 21,22, 3,27,64, 12,40,60,
                   5,42, 6,22, 160, 2]
    },
    {
        'args': '병렬 데이터라고 하면 앞서 수행한 태깅 작업의 데이터를 생각할 수 있지만,',
        'return': [0, 10,28,65, 8,28,51, 165, 6,27, 14,42, 19,26, 8,22, 3,30, 165, 21,22, 9,28,46, 165, 14,22,71, 12,26, 165, 12,35,
                   21,23,65, 21,22,46, 165, 19,23, 3,42,65, 165, 15,22,43, 14,26,60, 14,41, 165, 6,27, 14,42, 19,26, 8,40,51, 165,
                   12,23,65, 3,22,43, 21,22,51, 165, 12,35, 165, 14,42,64, 15,42, 9,22,46, 159, 2]
    },
    {
        'args': 'tar는 target의 줄임말로 번역하고자 하는 문장을 나타냅니다.',
        'return': [0, 92, 73, 90, 5,40,46, 165, 92, 73, 90, 79, 77, 92, 14,41, 165, 15,35,51, 14,42,59, 9,22,51, 8,30, 165, 10,26,46, 14,28,
                   43, 21,22, 3,30, 15,22, 165, 21,22, 5,40,46, 165, 9,35,46, 15,22,65, 14,40,51, 165, 5,22, 19,22, 5,23,60, 5,42, 6,22,
                   160, 2]
    },
]

test_sheet_split_nums = [
{
        'args': [0, 14,42, 15,27, 165, 91, 77, 89, 127, 91, 77, 89, 8,40,51, 165, 14,42, 14,34,65, 21,23, 12,26, 165, 3,42, 3,29, 165,
                10,26,46, 60, 14,28,43, 3,42, 8,40,51, 165, 9,22,46, 6,40,51, 14,26, 41, 10,30, 6,30, 8,30,43, 165, 21,22, 3,27,64, 12,40,60,
                5,42, 6,22, 160, 2],
        'return': [[0], [14,42], [15,27], [165], [91], [77], [89], [127], [91], [77], [89], [8,40,51], [165], [14,42], [14,34,65], [21,23], [12,26], [165], [3,42], [3,29], [165],
                   [10,26,46], [60], [14,28,43], [3,42], [8,40,51], [165], [9,22,46], [6,40,51], [14,26], [41], [10,30], [6,30], [8,30,43], [165], [21,22], [3,27,64], [12,40,60],
                   [5,42], [6,22], [160], [2]]
    },
    {
        'args': [0, 10,28,65, 8,28,51, 165, 6,27, 14,42, 19,26, 8,22, 3,30, 165, 21,22, 9,28,46, 165, 14,22,71, 12,26, 165, 12,35,
                 21,23,65, 21,22,46, 165, 19,23, 3,42,65, 165, 15,22,43, 14,26,60, 14,41, 165, 6,27, 14,42, 19,26, 8,40,51, 165,
                 12,23,65, 3,22,43, 21,22,51, 165, 12,35, 165, 14,42,64, 15,42, 9,22,46, 159, 2],
        'return': [[0], [10,28,65], [8,28,51], [165], [6,27], [14,42], [19,26], [8,22], [3,30], [165], [21,22], [9,28,46], [165], [14,22,71], [12,26], [165], [12,35],
                   [21,23,65], [21,22,46], [165], [19,23], [3,42,65], [165], [15,22,43], [14,26,60], [14,41], [165], [6,27], [14,42], [19,26], [8,40,51], [165],
                   [12,23,65], [3,22,43], [21,22,51], [165], [12,35], [165], [14,42,64], [15,42], [9,22,46], [159], [2]]
    },
    {
        'args': [0, 92, 73, 90, 5,40,46, 165, 92, 73, 90, 79, 77, 92, 14,41, 165, 15,35,51, 136, 14,42,59, 9,22,51, 8,30, 165, 10,26,46, 14,28,43,
                   21,22, 3,30, 15,22, 165, 21,22, 5,40,46, 165, 9,35,46, 15,22,65, 14,40,51, 165, 5,22, 19,22, 5,23,60, 5,42, 6,22,
                   160, 2],
        'return': [[0], [92], [73], [90], [5,40,46], [165], [92], [73], [90], [79], [77], [92], [14,41], [165], [15,35,51], [136], [14,42,59], [9,22,51], [8,30], [165], [10,26,46], [14,28,43],
                   [21,22], [3,30], [15,22], [165], [21,22], [5,40,46], [165], [9,35,46], [15,22,65], [14,40,51], [165], [5,22], [19,22], [5,23,60], [5,42], [6,22],
                   [160], [2]]
    },
]

class TextProcessing(unittest.TestCase):

    def test_token_to_num(self):
        for test_item in test_sheet_token_to_num:
            args = test_item['args']
            truth = test_item['return']
            ret = text_processing.token_to_num(args)
            self.assertTrue(functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, truth, ret), True), f'test_sheet: {args}:{truth}, function return: {ret}')


    def test_num_to_token(self):
        for test_item in test_sheet_token_to_num:
            args = test_item['return']
            truth = test_item['args']
            ret = text_processing.num_to_token(args)
            self.assertTrue(functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, truth, ret), True), f'test_sheet: {args}:{truth}, function return: {ret}')

    def test_isHangul(self):
        for test_item in test_sheet_is_hangul:
            args = test_item['args']
            truth = test_item['return']
            ret = text_processing.isHangul(args)
            self.assertEqual(truth,ret)

    def test_tokenize_text(self):
        for test_item in test_sheet_tokenize_text:
            args = test_item['args']
            truth = test_item['return']
            ret = text_processing.tokenize_text(args)
            self.assertTrue(functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, truth, ret), True), f'test_sheet: {args}:{truth}, function return: {ret}')

    def test_split_nums(self):
        for test_item in test_sheet_split_nums:
            args = test_item['args']
            truth = test_item['return']
            ret = text_processing.split_nums(args)
            self.assertTrue(functools.reduce(lambda x, y: x and y,
                                             map(lambda truth_splited, ret_splited:
                                                 functools.reduce(lambda x, y: x and y,
                                                                  map(lambda p, q: p == q,
                                                                      truth_splited, ret_splited), True),
                                                 truth, ret), True),
                            f'test_sheet: {args}:{truth}, function return: {ret}')

    def test_text_to_nums(self):
        for test_item in test_sheet_text_to_nums:
            args = test_item['args']
            truth = test_item['return']
            ret = text_processing.text_to_nums(args)
            self.assertTrue(functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, truth, ret), True), f'test_sheet: {args}:{truth}, function return: {ret}')

    def test_nums_to_text(self):
        for test_item in test_sheet_text_to_nums:
            args = test_item['return']
            truth = test_item['args']
            ret = text_processing.nums_to_text(args)
            self.assertEqual(truth, ret)

if __name__ == '__main__':
    unittest.main()