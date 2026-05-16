from typing import Any

import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import joblib
import pandas as pd

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(layout="wide", page_title="Доступная среда AI")

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ---
CITIES = {
    "Москва": {"coords": [55.7512, 37.6184], "zoom": 12},
    "Санкт-Петербург": {"coords": [59.9343, 30.3351], "zoom": 12}
}

if 'current_city_name' not in st.session_state:
    st.session_state.current_city_name = "Москва"
if 'map_center' not in st.session_state:
    st.session_state.map_center = CITIES["Москва"]["coords"]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = CITIES["Москва"]["zoom"]
if 'start_point' not in st.session_state:
    st.session_state.start_point = {"coords": None, "address": ""}
if 'end_point' not in st.session_state:
    st.session_state.end_point = {"coords": None, "address": ""}
if 'route_stats' not in st.session_state:
    st.session_state.route_stats = None

# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_resources() -> tuple[Any, Nominatim]:
    model = joblib.load('city_access_model.pkl')
    geolocator = Nominatim(user_agent="city_access_final")
    return model, geolocator

model, geolocator = load_resources()

# --- ФУНКЦИИ ПОМОЩНИКИ---
def update_address(point_key, coords) -> None:
    try:
        location = geolocator.reverse(coords)
        if location:
            st.session_state[point_key]["address"] = location.address
            st.session_state[point_key]["coords"] = coords
    except:
        pass

def update_coords(point_key, address) -> None:
    try:
        location = geolocator.geocode(address)
        if location:
            st.session_state[point_key]["coords"] = (location.latitude, location.longitude)
            st.session_state[point_key]["address"] = address
    except:
        st.error("Адрес не найден")

def reset_map() -> None:
    city = st.session_state.get("current_city_name", "Москва")
    st.session_state.start_point = {"coords": None, "address": ""}
    st.session_state.end_point = {"coords": None, "address": ""}
    st.session_state.map_center = CITIES[city]["coords"]
    st.session_state.map_zoom = CITIES[city]["zoom"]

    if 'current_route_gdf' in st.session_state:
        del st.session_state['current_route_gdf']
    st.session_state.route_stats = None
    if 'last_click_processed' in st.session_state:
        del st.session_state['last_click_processed']
    st.session_state.needs_fit_bounds = False

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.title("🧭 Навигация")
selected_city = st.sidebar.selectbox("Выберите город:", options=list(CITIES.keys()), key="city_sel")

if st.session_state.current_city_name != selected_city:
    st.session_state.current_city_name = selected_city
    reset_map()
    st.rerun()

start_addr = st.sidebar.text_input("Точка А (Откуда):", value=st.session_state.start_point["address"])
if st.sidebar.button("Найти А"):
    update_coords("start_point", start_addr)
    st.rerun()

end_addr = st.sidebar.text_input("Точка Б (Куда):", value=st.session_state.end_point["address"])
if st.sidebar.button("Найти Б"):
    update_coords("end_point", end_addr)
    st.rerun()

if st.sidebar.button("🧹 Очистить всё"):
    reset_map()
    st.rerun()

get_path = st.sidebar.button("🚀 Построить маршрут", type="primary")

# --- ИНФОРМАЦИОННОЕ ОКНО ---
with st.sidebar.expander("🤖 Как работает интеллект маршрута?"):
    st.write("""
    В основе проекта лежит модель **Random Forest**, обученная на 2500+ сегментах дорог Москвы и Санкт-Петербурга.

    **Как это работает:**
    1. **Анализ признаков:** Модель изучает тип покрытия (`surface`), категорию дороги (`highway`), наличие освещения (`lit`) и ровность (`smoothness`).
    2. **Предсказание:** Даже если в картах OSM не указан тег 'wheelchair' (доступность для людей с ограниченными возможностями), нейросеть предсказывает доступность на основе косвенных признаков с точностью **86.5%**.
    3. **Штрафные веса:** Участки, признанные "недоступными", получают 20-кратный штраф к "длине". Алгоритм поиска пути воспринимает 100 метров плохой дороги как 2 километра и ищет обход.
    4. **Цветовая индикация:**
        - 🟢 Зеленый — модель уверена в доступности.
        - 🟠 Оранжевый — путь проложен через сложный участок за неимением альтернатив.
    """)

# --- ЛОГИКА ПОСТРОЕНИЯ МАРШРУТА ---
if get_path:
    if st.session_state.start_point["coords"] and st.session_state.end_point["coords"]:
        with st.spinner("Анализ доступности..."):
            try:
                # Считаем расстояние и задаем динамический радиус для загрузки графа,чтобы не грузить
                # слишком большой кусок города, но при этом гарантировать покрытие маршрута
                p1, p2 = st.session_state.start_point["coords"], st.session_state.end_point["coords"]
                dist = geodesic(p1, p2).meters
                G = ox.graph_from_point(p1, dist=dist+500, network_type='walk')

                # Векторизованное предсказание доступности для всех рёбер графа
                edges_list = []
                for u, v, k, data in G.edges(data=True, keys=True):
                    edges_list.append(
                        {
                            'highway': str(data.get('highway', 'unknown')),
                            'surface': str(data.get('surface', 'unknown')),
                            'smoothness': str(data.get('smoothness', 'unknown')),
                            'lit': str(data.get('lit', 'unknown')),
                            'length': float(data.get('length', 0)), 'id': (u, v, k)
                        }
                    )
                df_edges = pd.DataFrame(edges_list)
                preds = model.predict(df_edges.drop(columns=['id']))

                # Записываем штрафные веса в граф для рёбер, которые модель считает недоступными
                for i, row in df_edges.iterrows():
                    u, v, k = row['id']
                    G[u][v][k]['access_weight'] = row['length'] * (20 if preds[i] == 0 else 1)

                # Поиск кратчайшего пути с учетом новых весов
                orig = ox.nearest_nodes(G, X=p1[1], Y=p1[0])
                dest = ox.nearest_nodes(G, X=p2[1], Y=p2[0])
                route = ox.shortest_path(G, orig, dest, weight='access_weight')

                if route:
                    # Получаем GeoDataFrame для маршрута и предсказываем доступность каждого сегмента
                    route_gdf = ox.routing.route_to_gdf(G, route)
                    r_features = pd.DataFrame(
                        [
                            {
                                'highway': str(r.get('highway','unknown')),
                                'surface': str(r.get('surface','unknown')),
                                'smoothness': str(r.get('smoothness','unknown')),
                                'lit': str(r.get('lit','unknown')),
                                'length': float(r.get('length',0))
                            } for _, r in route_gdf.iterrows()
                        ]
                    )

                    # Предсказываем доступность для каждого сегмента маршрута и сохраняем результат в GeoDataFrame
                    route_gdf['is_accessible'] = model.predict(r_features)
                    st.session_state.current_route_gdf = route_gdf
                    st.session_state.route_stats = {
                        "length": int(route_gdf['length'].sum()),
                        "access": (preds.sum()/len(preds))*100
                    }
                    st.session_state.needs_fit_bounds = True
                    st.rerun()

            except Exception as e:
                st.error(f"Ошибка: {e}")
    else:
        st.sidebar.warning("Выберите точки")

# --- ИНТЕРФЕЙС КАРТЫ ---
st.title("🏙️ Доступная городская среда")

# Метрики маршрута
if st.session_state.route_stats:
    c1, c2 = st.columns(2)
    c1.metric("📏 Длина маршрута", f"{st.session_state.route_stats['length']} м")
    c2.metric("♿ Доступность района", f"{st.session_state.route_stats['access']:.1f}%")

# Обновляем центр, если появилась новая точка, но ЕЩЕ НЕТ маршрута
if st.session_state.end_point["coords"] and not st.session_state.route_stats:
    st.session_state.map_center = st.session_state.end_point["coords"]
    st.session_state.map_zoom = 15
elif st.session_state.start_point["coords"] and not st.session_state.end_point["coords"]:
    st.session_state.map_center = st.session_state.start_point["coords"]
    st.session_state.map_zoom = 15

# Создание карты с текущим центром и зумом из состояния, с включенной шкалой контроля масштаба
m = folium.Map(
    location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, control_scale=True
)

# Отрисовка слоев (Маршрут и Маркеры) - сначала маршрут, чтобы маркеры были поверх линий
if 'current_route_gdf' in st.session_state:
    all_p = []
    for _, row in st.session_state.current_route_gdf.iterrows():
        if row.geometry.geom_type == 'LineString':
            line = [(c[1], c[0]) for c in row.geometry.coords]
            color = "#2e7d32" if row.get('is_accessible', 1) == 1 else "#ff9800"
            folium.PolyLine(line, color=color, weight=7, opacity=0.8).add_to(m)
            all_p.extend(line)
    if all_p and st.session_state.get('needs_fit_bounds'):
        m.fit_bounds(all_p)
        st.session_state.needs_fit_bounds = False

if st.session_state.start_point["coords"]:
    folium.Marker(st.session_state.start_point["coords"], icon=folium.Icon(color='green')).add_to(m)
if st.session_state.end_point["coords"]:
    folium.Marker(st.session_state.end_point["coords"], icon=folium.Icon(color='red')).add_to(m)

# Вывод карты (Минимальный набор объектов для возврата)
output = st_folium(m, width=1000, height=600, key="v_final", returned_objects=["last_clicked"])

# Логика кликов по карте (Если кликнули и это новый клик, то обновляем либо старт, либо энд по координатам клика и адресу)
if output and output.get("last_clicked"):
    click_c = [output["last_clicked"]["lat"], output["last_clicked"]["lng"]]
    if st.session_state.get('last_click_processed') != click_c:
        st.session_state.last_click_processed = click_c
        if not st.session_state.start_point["coords"]:
            update_address("start_point", click_c)
            st.rerun()
        elif not st.session_state.end_point["coords"]:
            update_address("end_point", click_c)
            st.rerun()
