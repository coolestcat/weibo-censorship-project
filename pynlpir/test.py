#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pynlpir

pynlpir.open()

# s = '在这种情况下，如果领导人高高在上，不解民间疾苦。老百姓就会仇恨当官的，就会给他们找麻烦，变成刁民。而在中央集权情况下，只有巴结上司和领导，才能往上爬，不能说领导人坏话。这就是底线，在这种“愚民统治”下，久而久之，人民也就变成了愚民。有话不敢说，有话不敢讲，或者讲出来的话带刺的愚民'
# segmented = pynlpir.segment(s)
# for segment in segmented:
# 	print segment[0] + "\t" + segment[1]

# print "------"

# s = 'Bill是美国总统,好像没来过北京理工大学,喜欢吃小尾羊'
# segmented = pynlpir.segment(s)
# for segment in segmented:
# 	print segment[0] + "\t" + segment[1]

n = pynlpir.nlpir.ImportUserDict("userdict.txt")
print "num imported: " + str(n)

# s = 'Bill是美国总统,好像没来过北京理工大学,喜欢吃小尾羊'
# segmented = pynlpir.segment(s)
# for segment in segmented:
# 	print segment[0] + "\t" + segment[1]

# s = '43年前，年仅16岁的习近平前往陕北插队，并在那里当了七年的农民，习近平在接受央视专访时曾谈起过那段岁月，他说："我几乎那一年365天没有歇着，除了生病。下雨刮风我在窑洞里跟他们铡草，晚上跟着看牲口，然后跟他们去放羊，什么活都干，因为我那个时候扛200斤麦子，十里山路我不换肩的。'
# segmented = pynlpir.segment(s)
# for segment in segmented:
# 	print segment[0] + "\t" + segment[1]

s = '现在我挺讨厌看到新闻压倒性反对难民来的声音，你们真以为小土豆是傻的吗…政治舞台上没有对错，只有利益。小土豆跟着美国混，连美国都收了1W个，小土豆能不收吗？跟着美国混的国家都收了，为什么？你们冲过去打人家就算了，还不收难民，谁不收以后这就是把柄。2W5这个数字也是在保守党和自由党中折衷出来的。但是延迟和加强审核，或出台强硬管理措施，我还是觉得非常占理，毕竟对当局来说，怎么样本国人命的性命也比他国人民更重要。'
segmented = pynlpir.segment(s)
for segment in segmented:
	print segment[0] + "\t" + segment[1]