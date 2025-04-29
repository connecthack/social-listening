import pandas as pd
import json
import threading
import requests
import time
from datetime import datetime, timezone
import concurrent.futures
import schedule
from openai import OpenAI
import numpy as np
import tiktoken
import dotenv
import os

from fill_database import Database

dotenv.load_dotenv('../.env')

OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')  
TOKEN_1 = os.getenv('TOKEN_1')  
TOKEN_2 = os.getenv('TOKEN_2')  

# Функция для получения эмбеддингов
def get_embedding(texts, model="text-embedding-3-small"):
    texts = [text.replace("\n", " ").strip() for text in texts]
    texts = [t if len(t) > 0 else "none" for t in texts]

    embeddings = []
    client = OpenAI(api_key=OPENAI_APIKEY)
    data = client.embeddings.create(input=texts, model=model).data
    embeddings = [data[i].embedding for i in range(len(data))]

    for i, e in enumerate(embeddings):
        embeddings[i] = [np.round(e1, 8) for e1 in e]

    return embeddings

# Функция для обрезки текста до лимита токенов
def truncate_texts(texts, max_tokens=8000, model="text-embedding-3-small"):
    tokenizer = tiktoken.encoding_for_model(model)
    truncated_texts = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        truncated_texts.append(tokenizer.decode(tokens))
    return truncated_texts

# Получение постов с фильтрацией по дате
def GetPosts(owner_id, offset, token, last_date):
    last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
    try:
        url = f'https://api.vk.com/method/wall.get?owner_id={owner_id}&count=100&offset={offset}&access_token={token}&v=5.103'
        response_owner = requests.get(url, timeout=5)

        if 'available only for community members' not in str(response_owner.json()) and 'wall is disabled' not in str(
                response_owner.json()):
            items = response_owner.json().get('response', {}).get('items', [])
            filtered_items = [
                post for post in items if datetime.fromtimestamp(post['date']) >= last_date_obj
            ]
            return filtered_items
        else:
            print(f'Паблик {owner_id} недоступен')
            return []
    except Exception as e:
        print(f'Ошибка при запросе: {e}')
        time.sleep(600)
        return []

# Запись постов в файл и обработка эмбеддингов
def WriteToDB(data, owner_id):
    global embed_batch  # Глобальная переменная для накопления батчей
    count = 0
    data_to_write = list()
    if data:
        for item in data:
            res = {
                'public_id': owner_id,
                'post_text': item.get('text', None),
                'post_date': datetime.fromtimestamp(item['date'], tz=timezone.utc),
                'post_id': item.get('id'),
                'views': item.get('views', {}).get('count', 0),
                'reposts': item.get('reposts', {}).get('count', 0),
                'likes': item.get('likes', {}).get('count', 0),
                'comments': item.get('comments', {}).get('count', 0)
            }
            data_to_write.append([v for v in res.values()])

            # Добавляем данные в батч для эмбеддингов
            if res['post_text']:
                embed_batch.append({
                    'PostText': res['post_text'],
                    'PostID': res['post_id'],
                    'OwnerID': res['public_id']
                })

            # Проверяем размер батча
            if len(embed_batch) >= 1000:
                ProcessEmbeddings()  # Обрабатываем эмбеддинги
            count += 1

    db.add_posts(data_to_write)

    return count

# Обработка эмбеддингов и сохранение в файл
def ProcessEmbeddings():
    global embed_batch
    if embed_batch:
        texts = [item['PostText'] for item in embed_batch]
        embeddings = get_embedding(texts)

        data = [[embed_batch[i]['PostID'], embeddings[i]] for i in range(len(embed_batch))]
        db.add_embeddings(data)

        # Очищаем батч
        embed_batch = []

# Обработка пабликов с использованием токена
def ProcessWithToken(ids_and_dates, token, token_id):
    for owner_id, last_date in ids_and_dates:
        print(f"Обрабатываем ID: {owner_id}, Дата последнего поста: {last_date} с токеном {token_id}")
        offset = 0
        counter = 0

        posts = GetPosts(owner_id, offset, token, last_date)
        if posts:
            id_last_date.append([owner_id, posts[0]['date']])
            counter += WriteToDB(posts, owner_id)

            while len(posts) == 100:
                offset += 100
                posts = GetPosts(owner_id, offset, token, last_date)
                if not posts:
                    break
                counter += WriteToDB(posts, owner_id)

        print(f"Всего обработано {counter} постов для ID: {owner_id}")

# Основная функция для запуска программы
def ScrappingPosts():
    ids_and_dates = db.get_last_upds()

    mid = len(ids_and_dates) // 2
    ids_token1 = ids_and_dates[:mid]
    ids_token2 = ids_and_dates[mid:]

    print(f"Часть 1: {len(ids_token1)} ID, Часть 2: {len(ids_token2)} ID")

    token1 = TOKEN_1
    token2 = TOKEN_2

    ProcessWithToken(ids_token1, token1, 1)
    ProcessWithToken(ids_token2, token2, 2)

    # TODO: многопоточность
    # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #     future1 = executor.submit(ProcessWithToken, ids_token1, token1, 1)
    #     future2 = executor.submit(ProcessWithToken, ids_token2, token2, 2)

    #     concurrent.futures.wait([future1, future2])

    db.add_last_upds(id_last_date)
    ProcessEmbeddings()  # Обрабатываем оставшиеся эмбеддинги

# Главная функция
def main():
    schedule.every().day.at("00:00").do(ScrappingPosts)
    print("Программа запущена. Выполняем первый запуск...")
    ScrappingPosts()
    print("Ожидаем следующего запуска...")
    while True:
        schedule.run_pending()
        time.sleep(1)

# Глобальные переменные
write_lock = threading.Lock()
id_last_date = []
embed_batch = []  # Батч для эмбеддингов
db = Database()

if __name__ == "__main__":
    main()