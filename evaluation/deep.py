import stanza

# Initialize the pipeline
nlp = stanza.Pipeline(lang='zh')

# Input Chinese sentence

# article = '''“爸爸，什么时候才会下雪呢?”小熊问。\n“可能还要等很久吧，”
#             熊爸爸说，“一直到我们睡着了以后。”\n“等我们睡醒了，雪还会在吗?”小熊又问。
#             \n“那时，春天就来了，雪会融化，渗进泥土里。”熊爸爸说。\n好想看一看雪啊，这是小熊心中藏了很久很久的愿望。
#             \n熊爸爸把小熊举起来，抛在厚厚的床垫上。
#             \n“嘿，是不是很舒服?”熊爸爸说，“我在枕头下面放了香香的花朵和干果，不知道它们会不会跑进你的梦里。”
#             \n“可是爸爸，我不想睡觉，我要等着看雪。”小熊说。\n熊爸爸把小熊抱在怀里，让小熊觉得好温暖。
#             \n“天气会越来越冷，你一定会非常怕冷。”熊爸爸说。\n小熊看了看窗外——
#             \n风使劲吹着，黄色的叶子飞起来、又落下去，看上去真的很冷。
#             \n而且，小熊也有一点点困了。
#             \n“所以，还是好好地睡上一觉吧，睡到太阳出来、树林变得暖和的时候。”熊爸爸说。
#             \n熊妈妈做了很多苹果酱，涂在小熊的手掌上。
#             \n“当你肚子饿得咕咕叫时，就舔舔你的手掌。”熊妈妈说，“吃饱了，再呼呼睡个好觉。”
#             \n“我醒来的时候，会先舔舔果酱，然后再去看看下雪。”小熊说。
#             \n“那时你会觉得特别困，只想着赶快睡觉。”熊妈妈笑起来。\n“不"'''
# f=open('test_human.txt','r',encoding="utf-8")
# f=open('test_cpm.txt','r',encoding="utf-8")
# f=open('test_cpm_all.txt','r',encoding="utf-8")
# f=open('test_pangu.txt','r',encoding="utf-8")
f=open('/data2/data/wxx/LLaMA-Factory/data/ccstories/result_qwen3_0.6b-ft-4000.txt','r',encoding="utf-8")
# f=open('llm/result_llama-chinese-ft.txt','r',encoding="utf-8")
depth_article_avg = 0
dependency_count_article_avg = 0
for i, article in enumerate(f.readlines()):
    # if i % 2 == 0:
    #     continue
    if len(article) < 2:
        continue
    doc = nlp(article.replace("\\n",""))
    depth_article = 0
    dependency_count_article = 0
    print(f"Article {i + 1}: {doc.text}")
    if len(doc.sentences) == 0:
        continue
    for i, sentence in enumerate(doc.sentences):
        if len(sentence.text) < 3:
            continue
        # print(f"Sentence {i + 1}: {sentence.text}")
        # Get the first sentence (since `doc` contains a list of sentences)
        sentence_obj = sentence

        # Print the sentence's dependency tree
        # for word in sentence_obj.words:
            # print(f"Word: {word.text}, Head: {word.head}, Deprel: {word.deprel}")

        # Calculate tree depth by counting levels of dependency
        def calculate_tree_depth(sentence_obj):
            # Create a dictionary of heads for each word
            head_dict = {word.id: word.head for word in sentence_obj.words}
            
            # Function to calculate depth from a node (word)
            def depth(word_id):
                if word_id == 0:  # root has no head (head = 0)
                    return 1
                return 1 + depth(head_dict[word_id])

            # Find depth for each word and return the maximum depth
            return max(depth(word.id) for word in sentence_obj.words)

        # Get the tree depth
        depth = calculate_tree_depth(sentence_obj)
        # print(f"Sentence Tree Depth: {depth}")
        depth_article += depth

        # Calculate the number of dependencies (dependency count)
        dependency_count = len(sentence_obj.words)
        # print(f"Dependency Count: {dependency_count}")
        dependency_count_article += dependency_count
    depth_article_avg += depth_article/len(doc.sentences)
    dependency_count_article_avg += dependency_count_article/len(doc.sentences)
depth_dataset_avg = depth_article_avg/1000
dependency_count_dataset_avg = dependency_count_article_avg/1000

print(f"Sentence Tree Depth: {depth_dataset_avg}")
print(f"Dependency Count: {dependency_count_dataset_avg}")