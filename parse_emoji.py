import emoji
import functools
import operator

em = '🌞🧔\u200d♂️'
# em = u'\\ud83c'
print(em)
# em = 'Hey 👨‍👩‍👧‍👧👨🏿😷😷🇬🇧'
em_split_emoji = emoji.get_emoji_regexp().split(em)
em_split_whitespace = [substr.split() for substr in em_split_emoji]
em_split = functools.reduce(operator.concat, em_split_whitespace)

e = []
for separated in em_split:
    print(separated)
    separated = separated.replace('\u200d', ' ').split()
    e += [emoji.demojize(s) for s in separated]
print(e)

print(bytes('test \\u0d83d \\u0259', 'utf-8').decode('unicode-escape'))

