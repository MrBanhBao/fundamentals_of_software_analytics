#!/usr/bin/env python

import sys

data = sys.stdin.read()
author_dict = {}
author_email_set = set(data.split("\n"))

for author_email in author_email_set:
    try:
        author = author_email.split(',')[0]
        email = author_email.split(',')[1]

        if author in author_dict:
            author_dict[author].append(email)
        else:
            author_dict[author] = []
            author_dict[author].append(email)
    except IndexError:
        print('IndexError with str:', author_email)


for author in author_dict.keys():
    out_str = str(author)
    for email in author_dict[author]:
        out_str = out_str + ' ' + email

    print(out_str)
