import json
import os
import re
import requests
from bs4 import BeautifulSoup


def get_patent(num, details=True):
    s = requests.Session()
    search_string = num
    search_url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch' \
                 '-bool.html&r=1&f=G&l=50&co1=AND&d=PTXT&s1=' + search_string
    r = s.get(search_url)
    text = r.text
    print(num + ' finish collecting html...')
    soup = BeautifulSoup(text, 'html.parser')
    patent_data = dict()
    patent_data['id'] = num
    if details:
        patent_data['patent_code'] = soup.find('title').next[22:]
        patent_data['patent_name'] = soup.find('font', size='+1').text[:-1]
        tmp1 = text[re.search('BUF7=', text).span()[1]:]
        patent_data['year'] = tmp1[:re.search('\n', tmp1).span()[0]]
        patent_data['inventor_and_country_data'] = soup.find_all('table', width='100%')[2].contents[1].text
        tmp1 = text[re.search('Description', text).span()[1]:]
        tmp2 = tmp1[re.search('<HR>', tmp1).span()[1]:]
        patent_data['description'] = tmp2[
                                     re.search('<BR>', tmp2).span()[0]:(re.search('<CENTER>', tmp2).span()[0] - 9)]. \
            replace('<BR><BR> ', '')
        patent_data['application_number'] = re.sub('[^0-9]', '',
                                                   soup.find(string=re.compile('Appl. No.:')).parent.next_sibling.text)
        patent_data['abstract'] = soup.find(string=re.compile('Abstract')).parent.findNext('p').text

    patent_data['citations'] = []
    refs = soup.findAll('a', href=re.compile('netacgi/nph-Parser'))
    for ref in refs:
        tmp_number = re.sub('[^0-9]', '', ref.text)
        if len(tmp_number):
            patent_data['citations'].append(tmp_number)

    patent_data['related'] = []
    related = soup.find('b', string='Related U.S. Patent Documents')
    if related:
        for idx, val in enumerate(related.parent.findNext('table').findAll('td')):
            if idx % 5 == 3:
                patent_data['related'].append(re.sub('[^0-9]', '', val.text))
    return patent_data


if __name__ == '__main__':
    dataset_dir = 'patent-data'
    patent_dirs = os.listdir('patent-data')
    data_all = []
    for field in patent_dirs:
        path = dataset_dir + '/' + field
        files = os.listdir(path)
        data = []
        for file in files:
            data.append(get_patent(re.findall('\d+', file)[0]))
        data_all.extend(data)
        with open('patents ' + field + '.json', 'w') as f:
            json.dump(data, f)
    with open('patents_all.json', 'w') as f:
        json.dump(data_all, f)
