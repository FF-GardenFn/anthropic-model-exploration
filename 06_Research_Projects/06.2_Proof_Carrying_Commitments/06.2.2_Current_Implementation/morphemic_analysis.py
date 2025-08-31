def split_morphemes(word):
    parts = []
    for p in ['un', 're', 'dis']:
        if word.startswith(p):
            parts.append(p)
            word = word[len(p):]
            break
    parts.append(word)
    return parts


def pole_score(word):
    return float(len([c for c in word if c in 'aeiou']))