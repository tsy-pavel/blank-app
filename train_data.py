import osmnx as ox
import pandas as pd

# Списки районов для сбора
moscow_districts = [
    "ЦАО, Москва", "САО, Москва", "СВАО, Москва", "ВАО, Москва",
    "ЮВАО, Москва", "ЮАО, Москва", "ЮЗАО, Москва", "ЗАО, Москва", "СЗАО, Москва"
]

spb_districts = [
    "Центральный район, Санкт-Петербург", "Адмиралтейский район, Санкт-Петербург",
    "Василеостровский район, Санкт-Петербург", "Выборгский район, Санкт-Петербург",
    "Калининский район, Санкт-Петербург", "Кировский район, Санкт-Петербург",
    "Красногвардейский район, Санкт-Петербург", "Красносельский район, Санкт-Петербург",
    "Московский район, Санкт-Петербург", "Невский район, Санкт-Петербург",
    "Петроградский район, Санкт-Петербург", "Приморский район, Санкт-Петербург",
    "Фрунзенский район, Санкт-Петербург"
]

def collect_city_data(city_name, districts_list, filename):
    all_data = []
    ox.settings.useful_tags_way = [
        'highway', 'surface', 'width', 'lit', 'incline',
        'smoothness', 'sidewalk', 'wheelchair', 'footway', 'kerb'
    ]

    print(f"\n--- Начинаю сбор данных для: {city_name} ---")

    for area in districts_list:
        try:
            print(f"Загрузка района: {area}...")
            # Загружаем пешеходный граф
            G = ox.graph_from_place(area, network_type='walk')
            _, edges = ox.graph_to_gdfs(G)

            if 'wheelchair' in edges.columns:
                # Берем только те сегменты, где заполнена доступность для обучения
                tagged = edges.dropna(subset=['wheelchair']).copy()

                # Отбираем колонки, которые пригодятся для ML
                features = ['length', 'highway', 'surface', 'smoothness', 'lit', 'wheelchair']
                existing_features = [f for f in features if f in tagged.columns]

                all_data.append(tagged[existing_features])
                print(f"✅ Успех: {len(tagged)} сегментов с метками")
            else:
                print(f"ℹ️ В районе {area} нет размеченных данных wheelchair")

        except Exception as e:
            print(f"⚠️ Ошибка в {area}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        # Создаем целевой признак (1 - доступно, 0 - недоступно)
        final_df['target'] = final_df['wheelchair'].apply(
            lambda x: 1 if x in ['yes', 'designated'] else 0
        )

        # Предварительная очистка: заполняем пустые значения строк
        text_cols = ['surface', 'smoothness', 'highway', 'lit']
        for col in text_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).fillna('unknown')

        final_df.to_csv(filename, index=False)
        print(f"🚀 Готово! Датасет {city_name} сохранен в {filename}. Строк: {len(final_df)}")
        return final_df
    else:
        print(f"❌ Не удалось собрать данные для {city_name}")
        return None

# Запуск сбора для Москвы
df_moscow = collect_city_data("Москва", moscow_districts, "train_data_moscow.csv")

# Запуск сбора для Санкт-Петербурга
df_spb = collect_city_data("Санкт-Петербург", spb_districts, "train_data_spb.csv")

if df_moscow is not None and df_spb is not None:
    print("\nВсе данные успешно собраны в два разных файла!")