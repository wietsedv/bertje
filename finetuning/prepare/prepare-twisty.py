import os
import json
import random

src_path = 'data/original/twisty'
meta_path = os.path.join(src_path, 'TwiSty-NL.json')
data_dir = os.path.join(src_path, 'users_id')

out_dir = 'data/prepared/twisty'

num_tweets = 200


def tweet_text(tweet, replace_ents=True):
    text = tweet['text']

    if replace_ents:
        ents = []
        for tag in tweet['entities']['hashtags']:
            ents.append((tuple(tag['indices']), tag['text']))
        for url in tweet['entities']['urls']:
            ents.append((tuple(url['indices']), 'URL'))  # url['display_url']
        for user in tweet['entities']['user_mentions']:
            ents.append((tuple(user['indices']), user['name']))

        offset = 0
        for (i, j), txt in sorted(ents):
            if i + offset <= 0:
                text = text[j + offset + 1:]
                offset = - (j + 1)
                continue
            text = text[:i + offset] + txt + text[j + offset:]
            offset += len(txt) - (j - i)

    return text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')


def get_user_tweet_texts(user_id, tweet_ids):
    user_path = os.path.join(data_dir, '{}.json'.format(user_id))
    with open(user_path) as f:
        tweet_data = json.load(f)['tweets']

    user_tweets = []
    for tweet_id in tweet_ids:
        if tweet_id not in tweet_data:
            print('skipping tweet id "{}"'.format(tweet_id))
            continue
        user_tweets.append(tweet_text(tweet_data[tweet_id]))
    return user_tweets


def save(data, path, filename):
    with open(os.path.join(path, filename), 'w') as f:
        for user_data in data:
            for item in user_data:
                if len(item) > 0:
                    f.write('\t'.join(item) + '\n')


def main():
    with open(meta_path) as f:
        meta = json.load(f)

    random.seed(7684)

    dev1, dev2 = [], []
    data = []
    for user_id in meta.keys():
        user = meta[user_id]
        tweet_ids = user['confirmed_tweet_ids']
        tweet_texts = get_user_tweet_texts(user_id, tweet_ids)

        random.shuffle(tweet_texts)

        user_data = [(txt, user['gender'], user_id) for txt in tweet_texts]

        if len(user_data) < num_tweets:
            dev1.append(user_data)
            continue

        if len(user_data) > num_tweets:
            dev2.append(user_data[num_tweets:])

        data.append(user_data[:num_tweets])

    random.shuffle(data)

    fold_size = len(data) // 10

    test = data[:fold_size]
    train = data[fold_size:]

    os.makedirs(out_dir, exist_ok=True)

    print('train={} dev1={} dev2={} test={}'.format(len(train), len(dev1), len(dev2), len(test)))

    save(train, out_dir, 'train.tsv')
    save(test, out_dir, 'test.tsv')
    save(dev1, out_dir, 'dev.tsv')
    save(dev2, out_dir, 'dev2.tsv')


if __name__ == '__main__':
    main()
