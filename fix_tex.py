path = 'Ksaai.tex'
txt = open(path, encoding='utf-8').read()

fixes = [
    # Item 4 — xi_n exclusive feature count: 5 -> 7 (range 6-8)
    (
        'the 5 $\\xi_n$-exclusive descriptors',
        'the 7 $\\xi_n$-exclusive descriptors (range 6--8)'
    ),
    # Item 3 — Fig 2/4 total union: 31 -> 34
    (
        'The complete 31-feature matrix',
        'The complete 34-feature matrix'
    ),
    (
        'all 31 descriptors appearing',
        'all 34 descriptors appearing'
    ),
    (
        'for all 31 descriptors',
        'for all 34 descriptors'
    ),
]

for old, new in fixes:
    if old in txt:
        txt = txt.replace(old, new)
        print(f'  OK : {old[:60]}')
    else:
        print(f'  MISS: {old[:60]}')

open(path, 'w', encoding='utf-8').write(txt)
print('Done.')
