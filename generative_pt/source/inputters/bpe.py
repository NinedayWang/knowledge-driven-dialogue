#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from bpe_encoder import Encoder
import re

NUM = "<num>"

# Generated with http://pythonpsum.com
# test_corpus = '''
#     Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
#     Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
#     Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
#     Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
#     Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
# '''
test_corpus = []
with open("../../data/demo.train", "r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip("\n")
        ln = re.sub('\d+', NUM, ln).lower()
        test_corpus.append(ln)

encoder = Encoder(40000, 0.25)  # params chosen for demonstration purposes
encoder.fit(test_corpus)

example = "START person_topic_a video_topic_b video_topic_b 主演 person_topic_a person_topic_a 祖籍 美国 肯塔基州 欧 温斯波洛 person_topic_a 代表作 魔法黑森林 person_topic_a 好友 阿尔 · 帕西诺 person_topic_a 性别 男 person_topic_a 职业 导演 person_topic_a 领域 明星 video_topic_b 时光网 短评 奇幻 够 奇幻 ！ 但 平淡 如 水 的 剧本 靠 德普 和 海瑟薇 也 无法 撑 起 ！ video_topic_b 主演 person_topic_a video_topic_b 时光网 评分 6.9 video_topic_b 类型 奇幻 video_topic_b 领域 电影 person_topic_a 评论 地球 上 最 帅 的 男人 之 一 ， 可惜 已经 沦为 仆街 三侠 。 。 。 person_topic_a 职业 经历 1997 , 德普演艺生涯 的 分水岭 person_topic_a 获奖 寻找梦幻岛_提名 _ ( 2005 ； 第7届 ) _ 青少年选择奖 _ 青少年选择奖 - 最佳 剧情 电影 男演员 : 突然 想 看 恐怖片 了 。 不 是 吧 ， 我 不敢 看 太 恐怖 的 。	person_topic_a 拍 过 一个 恐怖片 《 猛鬼街 》 ， 最近 要 上映 。	person_topic_a 祖籍 美国 肯塔基州 欧 温斯波洛person_topic_a 代表作 魔法黑森林person_topic_a 好友 阿尔 · 帕西诺person_topic_a 性别 男person_topic_a 职业 导演person_topic_a 领域 明星video_topic_b 时光网 短评 奇幻 够 奇幻 ！ 但 平淡 如 水 的 剧本 靠 德普 和 海瑟薇 也 无法 撑 起 ！video_topic_b 主演 person_topic_avideo_topic_b 时光网 评分 6.9video_topic_b 类型 奇幻video_topic_b 领域 电影person_topic_a 评论 地球 上 最 帅 的 男人 之 一 ， 可惜 已经 沦为 仆街 三侠 。 。 。person_topic_a 职业 经历 1997 , 德普演艺生涯 的 分水岭person_topic_a 获奖 寻找梦幻岛_提名 _ ( 2005 ； 第7届 ) _ 青少年选择奖 _ 青少年选择奖 - 最佳 剧情 电影 男演员"
example = re.sub('\d+', NUM, example).lower()

print(encoder.tokenize(example))

print(next(encoder.transform([example])))

print(next(encoder.inverse_transform(encoder.transform([example]))))

encoder.save("../../data/bpe")
print(len(encoder.vocabs_to_dict()['byte_pairs']))
print(len(encoder.vocabs_to_dict()['words']))
