import re, textwrap, json, sys
def first_two_players(anonymized, description):
    esc = re.escape(anonymized)
    # Replace escaped '[PLAYER] ([TEAM])'
    esc = re.sub(r'\\\[PLAYER\\\]\s*\\\(\[TEAM\\\]\\\)', r'(.+?)\\s*\\(.+?\\)', esc)
    # Replace standalone escaped '[PLAYER]'
    esc = re.sub(r'\\\[PLAYER\\\]', r'(.+?)', esc)
    # Replace other escaped placeholders
    esc = re.sub(r'\\\[[A-Z]+\\\]', r'.+?', esc)
    pattern = esc
    m = re.match(pattern, description)
    if not m:
        return ('nodata','nodata')
    groups=list(m.groups())[:2]
    while len(groups)<2:
        groups.append('nodata')
    return tuple(g.strip(' .') for g in groups)

anonymized="The substitution is prepared. [PLAYER] ([TEAM]) joins the action as a substitute, replacing [PLAYER]."
description="The substitution is prepared. Juan Manuel Mata (Manchester United) joins the action as a substitute, replacing Angel Di Maria."
print(first_two_players(anonymized, description))


# ---------- test ----------
anonymized = "[COACH] decides to make a substitution. [PLAYER] will be replaced by [PLAYER] ([TEAM])."
description  = "Jose Mourinho decides to make a substitution. Filipe Luis will be replaced by Didier Drogba (Chelsea)."

print(first_two_players(anonymized, description))
