from morphers.pos_tagger import POSTagger
import re
if __name__=="__main__":
    post = POSTagger()
    sent = "And if you threaten couch with legal action again, I 'll fuck you there, tie you up to your computer chair and go with me cradling your a** until it bleeds."
    modified_sentence = re.sub("\w*\*+\w*", "", sent)
    orig_parts, overall = post.generate(modified_sentence)

    fixed_parts: list[str] = []
    for val in orig_parts:
        if val.startswith("##"):
            fixed_parts[-1] = fixed_parts[-1][:-2] + val[2:]
        else:
            fixed_parts.append(val)

    parts = list(enumerate(fixed_parts))
    print(parts)