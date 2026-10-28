#!/usr/bin/env python
# coding: utf-8

# # ЛР2. Разведочный анализ и подготовка данных
# 
# **Датасет:** `RealEstate.zip` → `train.csv`. Это 10 000 квартир с характеристиками и ценой (`Price`).
# 
# **Цель работы:** освоить инструменты сбора и подготовки данных:
# 1. импортировать датасет;
# 2. найти пропуски и аномальные значения (статистикой и графиками), заполнить пропуски, устранить выбросы и проверить результат графиками;
# 3. сгенерировать новые признаки (не меньше двух) и закодировать категориальные признаки (OHE, бинаризация);
# 4. стандартизировать и нормализовать данные.
# 
# **Описание столбцов**
# 
# | Столбец | Смысл |
# |---|---|
# | `Id` | номер квартиры |
# | `DistrictId` | номер района |
# | `Rooms` | число комнат |
# | `Square`, `LifeSquare`, `KitchenSquare` | общая, жилая и кухонная площадь, м² |
# | `Floor`, `HouseFloor` | этаж квартиры и этажность дома |
# | `HouseYear` | год постройки дома |
# | `Ecology_1..3` | экологические показатели района (`_2`, `_3` категориальные: A/B) |
# | `Social_1..3` | социальные показатели района |
# | `Healthcare_1`, `Helthcare_2` | показатели медицины в районе |
# | `Shops_1`, `Shops_2` | магазины рядом (`Shops_2` категориальный: A/B) |
# | `Price` | цена квартиры (**целевая переменная**) |

# ## 0. Библиотеки
# 
# - **pandas**: таблица `DataFrame`, загрузка, фильтры, группировки;
# - **numpy**: математика над массивами;
# - **matplotlib / seaborn**: графики;
# - **scikit-learn**: готовые инструменты для кодирования и масштабирования признаков.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Binarizer

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

MAIN = "#3b6fb6"   # один цвет для всех «одиночных» графиков


# ## 1. Импорт датасета
# 
# `pd.read_csv` умеет читать CSV прямо из zip-архива, если в архиве один файл, поэтому распаковывать его не нужно.

# In[2]:


df = pd.read_csv("../RealEstate.zip")
raw = df.copy()          # исходная копия: пригодится для графиков «до / после»
print("Размер:", df.shape)
df.head()


# `df.info()` показывает тип каждого столбца и число **непустых** значений.
# Если непустых меньше 10 000, в столбце есть пропуски.

# In[3]:


df.info()


# `describe()` выводит основные статистики по числовым столбцам: среднее, стандартное отклонение, минимум, квартили и максимум.
# **Это первый инструмент поиска аномалий:** смотрим на `min` и `max` и спрашиваем себя, бывает ли такое в жизни.

# In[4]:


df.describe().T.round(2)


# **Что видно сразу:**
# - `HouseYear` max = **20 052 011**: опечатка, похоже на «2005–2011»;
# - `Rooms` min = 0 и max = 19: квартир без комнат не бывает, 19 комнат при такой площади тоже;
# - `Square` min ≈ 1 м², max ≈ 641 м²;
# - `LifeSquare` max ≈ 7 480 м², `KitchenSquare` max = 2 014 м² (похоже на год, записанный не в тот столбец);
# - `HouseFloor` min = 0 (дом без этажей) и max = 117;
# - у `LifeSquare` и `Healthcare_1` count меньше 10 000, значит, есть пропуски.

# ## 2. Пропуски и аномальные значения
# 
# ### 2.1 Пропуски
# `isna()` возвращает таблицу True/False (True означает пропуск), `.sum()` считает пропуски по столбцам.

# In[5]:


missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(1)
pd.DataFrame({"пропусков": missing, "%": missing_pct})[missing > 0]


# **Тепловая карта пропусков.** По горизонтали идут строки датасета, по вертикали столбцы, красные штрихи обозначают пропуски.
# Так сразу видно, в каких столбцах и насколько массово не хватает данных.

# In[6]:


fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(df.isna().T, cbar=False, cmap=["#e8edf5", "#d1495b"], ax=ax, yticklabels=True)
ax.set_xticks([])
ax.set_title("Пропуски в данных (красный: пропуск)")
ax.set_xlabel("строки датасета")
plt.tight_layout(); plt.show()


# Пропуски есть в двух столбцах:
# - `LifeSquare`: 21%. Жилую площадь можно разумно восстановить по общей площади;
# - `Healthcare_1`: 48%, почти половина. Это показатель района, поэтому логично заполнять его медианой по району.

# ### 2.2 Аномалии: статистические методы
# 
# **Метод межквартильного размаха (IQR).**
# Q1 и Q3 это 25-й и 75-й перцентили, IQR = Q3 − Q1.
# Значение считается выбросом, если оно меньше Q1 − 1.5·IQR или больше Q3 + 1.5·IQR.
# На этом же правиле построены «усы» BoxPlot.
# 
# **Метод Z-оценки.** z = (x − среднее) / стандартное отклонение, выбросом считается |z| > 3.
# Метод хорошо работает для нормально распределённых данных, но само среднее сильно сдвигается от выбросов,
# поэтому IQR обычно надёжнее.

# In[7]:


num_cols = ["Rooms", "Square", "LifeSquare", "KitchenSquare", "Floor", "HouseFloor",
            "HouseYear", "Social_3", "Healthcare_1", "Price"]

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

rows = []
for col in num_cols:
    s = df[col].dropna()
    lo, hi = iqr_bounds(s)
    z = (s - s.mean()) / s.std()
    rows.append({"признак": col, "нижняя граница IQR": round(lo, 1), "верхняя граница IQR": round(hi, 1),
                 "выбросов по IQR": int(((s < lo) | (s > hi)).sum()),
                 "выбросов по |z|>3": int((z.abs() > 3).sum())})
pd.DataFrame(rows)


# **Важно:** не всё, что IQR считает выбросом, является ошибкой.
# У площади и цены распределение скошено вправо: больших квартир меньше, но они настоящие.
# Если срезать всё выше Q3 + 1.5·IQR, мы выбросим реальные дорогие квартиры.
# Поэтому статистику дополняем **логическими (предметными) проверками**:

# In[8]:


checks = {
    "HouseYear > 2020 (год из будущего)":      (df.HouseYear > 2020).sum(),
    "Rooms == 0":                              (df.Rooms == 0).sum(),
    "Rooms > 5":                               (df.Rooms > 5).sum(),
    "Square < 15 м²":                          (df.Square < 15).sum(),
    "LifeSquare > Square":                     (df.LifeSquare > df.Square).sum(),
    "LifeSquare < 10 м²":                      (df.LifeSquare < 10).sum(),
    "KitchenSquare < 3 м² (не указана)":       (df.KitchenSquare < 3).sum(),
    "KitchenSquare > 30 м² или > Square/2":    ((df.KitchenSquare > 30) | (df.KitchenSquare > df.Square / 2)).sum(),
    "HouseFloor == 0":                         (df.HouseFloor == 0).sum(),
    "HouseFloor > 60":                         (df.HouseFloor > 60).sum(),
    "Floor > HouseFloor (этаж выше дома)":     (df.Floor > df.HouseFloor).sum(),
}
pd.Series(checks, name="строк").to_frame()


# ### 2.3 Аномалии: графики
# 
# **Гистограмма** показывает форму распределения. Длинный «хвост» или одиночный столбик далеко справа указывают на выбросы.

# In[9]:


cols = ["Square", "LifeSquare", "KitchenSquare", "Rooms", "HouseFloor", "Price"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), cols):
    sns.histplot(df[col].dropna(), bins=60, ax=ax, color=MAIN)
    ax.set_title(col)
plt.suptitle("Распределения ДО очистки", fontsize=14)
plt.tight_layout(); plt.show()


# **BoxPlot (ящик с усами).** Ящик занимает промежуток от Q1 до Q3, линия внутри это медиана, усы тянутся до 1.5·IQR.
# Точки за усами это кандидаты в выбросы. Видно, что у площадей несколько точек улетают в сотни и тысячи м²,
# а у `HouseYear` одна точка сжимает весь график.

# In[10]:


cols = ["Square", "LifeSquare", "KitchenSquare", "HouseFloor", "HouseYear", "Rooms"]
fig, axes = plt.subplots(1, 6, figsize=(16, 4.5))
for ax, col in zip(axes, cols):
    sns.boxplot(y=df[col], ax=ax, color=MAIN, flierprops={"markersize": 3})
    ax.set_title(col); ax.set_ylabel("")
plt.suptitle("BoxPlot ДО очистки", fontsize=14)
plt.tight_layout(); plt.show()


# **Диаграмма рассеяния** «общая площадь → жилая площадь». Все нормальные точки должны лежать **ниже** диагонали y = x,
# потому что жилая площадь не может превышать общую.

# In[11]:


fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df.Square, df.LifeSquare, s=6, alpha=0.4, color=MAIN)
lim = 300
ax.plot([0, lim], [0, lim], color="#555", lw=1.5, ls="--", label="LifeSquare = Square")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Square, м²"); ax.set_ylabel("LifeSquare, м²")
ax.set_title("Жилая площадь выше общей = ошибка (область над линией)")
ax.legend(); plt.tight_layout(); plt.show()


# ### 2.4 Заполнение пропусков и устранение выбросов
# 
# Принцип: **ошибочное значение превращаем в пропуск, а затем заполняем пропуск разумным значением.**
# Выбрасывать строки жалко, потому что в них остальные признаки нормальные.
# Для заполнения берём **медиану**: в отличие от среднего, она не сдвигается от выбросов.
# 
# Порядок важен. Сначала чиним «опорные» признаки (год, общая площадь, комнаты), потом зависящие от них (жилая площадь, кухня, этажи).
# 
# **1) HouseYear.** Год больше 2020 (данные собраны в 2020 году) это опечатка.
# `20052011` превращаем в 2005, `4968` в 1968 (первая цифра набрана неверно).
# Всё, что после этого осталось некорректным, заменяем медианой.

# In[12]:


df.loc[df.HouseYear == 20052011, "HouseYear"] = 2005
df.loc[df.HouseYear == 4968, "HouseYear"] = 1968
df.loc[df.HouseYear > 2020, "HouseYear"] = df.HouseYear.median()
df.HouseYear.describe()[["min", "max"]]


# **2) Square.** Квартира меньше 15 м² в этом датасете почти наверняка ошибка ввода.
# Заменяем такие значения медианой площади для квартир с тем же числом комнат.
# `groupby("Rooms").transform("median")` для каждой строки возвращает медиану её группы.

# In[13]:


df.loc[df.Square < 15, "Square"] = np.nan
df["Square"] = df.Square.fillna(df.groupby("Rooms").Square.transform("median"))
df["Square"] = df.Square.fillna(df.Square.median())   # на случай, если вся группа была пустой


# **3) Rooms.** 0 комнат или больше 5 при обычной площади это ошибка.
# Оцениваем число комнат по площади: делим площадь на типичную площадь одной комнаты и ограничиваем результат от 1 до 5.

# In[14]:


ok = df.Rooms.between(1, 5)
m2_per_room = (df.loc[ok, "Square"] / df.loc[ok, "Rooms"]).median()
print(f"Типичная площадь на одну комнату: {m2_per_room:.1f} м²")
bad = ~ok
df.loc[bad, "Rooms"] = (df.loc[bad, "Square"] / m2_per_room).round().clip(1, 5)
df.Rooms.value_counts().sort_index()


# **4) KitchenSquare.** Значения меньше 3 м² (часто 0 или 1) означают, что кухня не указана.
# Больше 30 м² или больше половины квартиры это ошибка. Такие значения заменяем медианой кухни по числу комнат.

# In[15]:


bad_k = (df.KitchenSquare < 3) | (df.KitchenSquare > 30) | (df.KitchenSquare > df.Square / 2)
df.loc[bad_k, "KitchenSquare"] = np.nan
df["KitchenSquare"] = df.KitchenSquare.fillna(df.groupby("Rooms").KitchenSquare.transform("median"))
print("Исправлено кухонь:", bad_k.sum())


# **5) LifeSquare.** Сначала помечаем ошибки как пропуски: жилая площадь больше общей, меньше 10 м²
# или вместе с кухней больше общей площади. Потом заполняем все пропуски (исходные и новые)
# через типичную **долю** жилой площади от общей: `LifeSquare = Square × медианная доля`.
# Это точнее, чем одна общая медиана, потому что у большой квартиры и жилая площадь большая.

# In[16]:


bad_l = (df.LifeSquare > df.Square) | (df.LifeSquare < 10) | (df.LifeSquare + df.KitchenSquare > df.Square)
df.loc[bad_l, "LifeSquare"] = np.nan
share = (df.LifeSquare / df.Square).median()
print(f"Медианная доля жилой площади: {share:.2f}")
df["LifeSquare"] = df.LifeSquare.fillna(df.Square * share)


# **6) HouseFloor и Floor.** Дом с 0 этажей и дома выше 60 этажей (в данных есть 117) считаем ошибкой
# и заменяем медианой. Если этаж квартиры оказался выше этажности дома, берём этажность равной этажу квартиры,
# потому что сам этаж квартиры указан надёжнее.

# In[17]:


df.loc[(df.HouseFloor == 0) | (df.HouseFloor > 60), "HouseFloor"] = np.nan
df["HouseFloor"] = df.HouseFloor.fillna(df.HouseFloor.median())
df["HouseFloor"] = np.maximum(df.HouseFloor, df.Floor)
print("Этаж выше дома:", (df.Floor > df.HouseFloor).sum())


# **7) Healthcare_1.** Пропущено 48%. Показатель относится к району, поэтому заполняем медианой **по району**
# (`DistrictId`). Если в районе вообще нет значений, берём общую медиану.
# Факт пропуска сам по себе может быть информативен, поэтому перед заполнением сохраним его
# в отдельный признак-флаг (он пригодится в п. 3).

# In[18]:


df["Healthcare_1_missing"] = df.Healthcare_1.isna().astype(int)
df["Healthcare_1"] = df.Healthcare_1.fillna(df.groupby("DistrictId").Healthcare_1.transform("median"))
df["Healthcare_1"] = df.Healthcare_1.fillna(df.Healthcare_1.median())


# **8) Ограничение хвостов (винзоризация).** После исправления явных ошибок у площадей ещё остаются
# редкие экстремальные значения. Мы не удаляем их, а «прижимаем» к 0.5-му и 99.5-му перцентилям методом `clip`.
# Порог выбран мягче IQR, чтобы не потерять настоящие большие квартиры.
# 
# Целевую переменную `Price` **не трогаем**: дорогие квартиры реальны, и модель должна учиться и на них.

# In[19]:


for col in ["Square", "LifeSquare", "KitchenSquare"]:
    lo, hi = df[col].quantile([0.005, 0.995])
    df[col] = df[col].clip(lo, hi)
    print(f"{col}: оставлен диапазон [{lo:.1f}; {hi:.1f}]")


# ### 2.5 Проверка: пропусков и аномалий не осталось

# In[20]:


print("Всего пропусков:", df.isna().sum().sum())
pd.Series({k: v for k, v in {
    "HouseYear > 2020": (df.HouseYear > 2020).sum(),
    "Rooms вне 1..5": (~df.Rooms.between(1, 5)).sum(),
    "Square < 15": (df.Square < 15).sum(),
    "LifeSquare > Square": (df.LifeSquare > df.Square).sum(),
    "KitchenSquare < 3": (df.KitchenSquare < 3).sum(),
    "HouseFloor == 0": (df.HouseFloor == 0).sum(),
    "Floor > HouseFloor": (df.Floor > df.HouseFloor).sum(),
}.items()}, name="строк").to_frame()


# In[21]:


fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(df.isna().T, cbar=False, cmap=["#e8edf5", "#d1495b"], ax=ax, yticklabels=True, vmin=0, vmax=1)
ax.set_xticks([])
ax.set_title("Пропуски ПОСЛЕ заполнения: карта однородная, пропусков нет")
plt.tight_layout(); plt.show()


# **BoxPlot до и после.** Слева исходные данные, справа очищенные. У каждого признака своя шкала:
# признаки сравниваются каждый сам с собой, а не друг с другом.

# In[22]:


cols = ["Square", "LifeSquare", "KitchenSquare", "HouseFloor", "HouseYear", "Rooms"]
fig, axes = plt.subplots(2, 6, figsize=(16, 8))
for i, col in enumerate(cols):
    sns.boxplot(y=raw[col], ax=axes[0, i], color="#9aa5b4", flierprops={"markersize": 3})
    sns.boxplot(y=df[col], ax=axes[1, i], color=MAIN, flierprops={"markersize": 3})
    axes[0, i].set_title(col); axes[0, i].set_ylabel(""); axes[1, i].set_ylabel("")
axes[0, 0].set_ylabel("ДО", fontsize=13); axes[1, 0].set_ylabel("ПОСЛЕ", fontsize=13)
plt.suptitle("BoxPlot до и после очистки", fontsize=14)
plt.tight_layout(); plt.show()


# In[23]:


cols = ["Square", "LifeSquare", "KitchenSquare", "Rooms", "HouseFloor", "HouseYear"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), cols):
    sns.histplot(df[col], bins=40, ax=ax, color=MAIN)
    ax.set_title(col)
plt.suptitle("Распределения ПОСЛЕ очистки", fontsize=14)
plt.tight_layout(); plt.show()


# In[24]:


fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df.Square, df.LifeSquare, s=6, alpha=0.4, color=MAIN)
lim = df.Square.max() * 1.05
ax.plot([0, lim], [0, lim], color="#555", lw=1.5, ls="--", label="LifeSquare = Square")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Square, м²"); ax.set_ylabel("LifeSquare, м²")
ax.set_title("ПОСЛЕ: все точки ниже диагонали"); ax.legend()
plt.tight_layout(); plt.show()


# **Как читать графики «после».** Точки за усами на нижнем ряду остались, и это нормально.
# BoxPlot по правилу 1.5·IQR помечает любые редкие значения, но теперь это реальные квартиры:
# площадь 100–130 м², 4–5 комнат, дома 1910-х годов и высотки до 48 этажей.
# Главное, что исчезли **невозможные** значения, из-за которых на верхнем ряду весь ящик сжимался в полоску.

# ### 2.6 Корреляции
# 
# Тепловая карта корреляций показывает, какие признаки связаны между собой и с ценой.
# Коэффициент корреляции Пирсона лежит в диапазоне от −1 до 1: около 0 связи нет, ближе к ±1 связь сильная линейная.

# In[25]:


corr = df.drop(columns=["Id"]).select_dtypes("number").corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, square=True, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Матрица корреляций (после очистки)")
plt.tight_layout(); plt.show()
corr["Price"].drop("Price").sort_values(ascending=False).round(2)


# Сильнее всего с ценой связаны площади и число комнат, что логично.
# `Social_1`, `Social_2` и `Social_3` сильно коррелируют друг с другом: они описывают одно и то же свойство района.

# ## 3. Новые признаки и кодирование категориальных
# 
# ### 3.1 Генерация новых признаков (feature engineering)
# Новые признаки дают модели информацию «в готовом виде», которую ей трудно извлечь самой:
# 
# | Новый признак | Формула | Зачем |
# |---|---|---|
# | `HouseAge` | 2020 − HouseYear | возраст дома понятнее года постройки |
# | `SquarePerRoom` | Square / Rooms | просторность квартиры |
# | `KitchenShare` | KitchenSquare / Square | доля кухни, признак планировки |
# | `RelativeFloor` | Floor / HouseFloor | положение по высоте: 0 внизу, 1 наверху |
# | `FloorCategory` | первый / последний / средний | первый и последний этажи традиционно дешевле |
# | `DistrictSize` | число квартир в районе | насколько район популярен (частотное кодирование `DistrictId`) |
# | `Healthcare_1_missing` | 1, если значение было пропущено | создан в п. 2.4 |

# In[26]:


df["HouseAge"] = 2020 - df.HouseYear
df["SquarePerRoom"] = df.Square / df.Rooms
df["KitchenShare"] = df.KitchenSquare / df.Square
df["RelativeFloor"] = df.Floor / df.HouseFloor

df["FloorCategory"] = np.select(
    [df.Floor == 1, df.Floor == df.HouseFloor],
    ["first", "last"],
    default="middle",
)

df["DistrictSize"] = df.groupby("DistrictId").Id.transform("count")

df[["HouseAge", "SquarePerRoom", "KitchenShare", "RelativeFloor", "FloorCategory", "DistrictSize"]].head()


# Проверим, что новые признаки действительно связаны с ценой.

# In[27]:


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
order = ["first", "middle", "last"]
sns.boxplot(data=df, x="FloorCategory", y="Price", order=order, ax=axes[0], color=MAIN, showfliers=False)
axes[0].set_title("Цена по категории этажа")
sns.scatterplot(data=df, x="SquarePerRoom", y="Price", s=6, alpha=0.3, ax=axes[1], color=MAIN, edgecolor=None)
axes[1].set_title("Цена от площади на комнату")
plt.tight_layout(); plt.show()
df.groupby("FloorCategory").Price.median().round(0).reindex(order)


# ### 3.2 Кодирование категориальных признаков
# 
# Модели машинного обучения работают с числами, поэтому текстовые категории нужно перевести в числа.
# 
# **Бинаризация.** У признака две категории, и они превращаются в 0 и 1.
# `Ecology_2`, `Ecology_3` и `Shops_2` принимают значения только A и B, поэтому кодируем `B → 1`, `A → 0`.

# In[28]:


for col in ["Ecology_2", "Ecology_3", "Shops_2"]:
    print(col, df[col].value_counts().to_dict())
    df[col] = df[col].map({"A": 0, "B": 1})
df[["Ecology_2", "Ecology_3", "Shops_2"]].head()


# **Бинаризация числового признака по порогу** (`Binarizer` из sklearn): значения выше порога становятся 1, остальные 0.
# Сделаем признак `IsNew`: 1, если дом построен после 2000 года (современный фонд), иначе 0.

# In[29]:


binarizer = Binarizer(threshold=2000)          # HouseYear > 2000 -> 1, иначе 0
df["IsNew"] = binarizer.fit_transform(df[["HouseYear"]]).astype(int).ravel()
df.IsNew.value_counts()


# **One-Hot Encoding (OHE).** Признак с k категориями превращается в k столбцов из 0 и 1, где в каждой строке ровно одна единица.
# Просто пронумеровать категории нельзя: first = 0, middle = 1, last = 2 навязало бы модели ложный порядок,
# как будто last «больше» first.
# 
# OHE применяем к `FloorCategory`. Показаны два способа: `pd.get_dummies` (pandas) и `OneHotEncoder` (sklearn).

# In[30]:


# Способ 1: pandas
dummies = pd.get_dummies(df.FloorCategory, prefix="Floor", dtype=int)
print(dummies.head())

# Способ 2: sklearn. Энкодер запоминает категории на обучающих данных
# и потом так же кодирует новые (тестовые) данные.
ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded = ohe.fit_transform(df[["FloorCategory"]])
print("Категории, найденные энкодером:", ohe.categories_[0])

df = pd.concat([df.drop(columns="FloorCategory"), dummies], axis=1)
df.filter(like="Floor").head()


# `DistrictId` тоже категориальный признак, хоть и записан числами: номер района ничего не говорит о его «величине».
# Но районов 205, и OHE дал бы 205 новых столбцов. Для такого признака удобнее **частотное кодирование**,
# его мы уже сделали в виде `DistrictSize`.

# ## 4. Стандартизация и нормализация
# 
# Признаки измеряются в разных масштабах: `Social_2` порядка тысяч, `KitchenShare` меньше 1.
# Многие алгоритмы (kNN, SVM, линейные модели с регуляризацией, нейросети) считают расстояния или градиенты,
# и признак с большими числами начинает доминировать просто из-за своего масштаба.
# 
# | Метод | Формула | Результат | Когда применять |
# |---|---|---|---|
# | **Стандартизация** (`StandardScaler`) | z = (x − mean) / std | среднее 0, стандартное отклонение 1 | по умолчанию; линейные модели, SVM, PCA |
# | **Нормализация** (`MinMaxScaler`) | x' = (x − min) / (max − min) | все значения в [0; 1] | нужен фиксированный диапазон: нейросети, kNN, изображения |
# 
# Что **не** масштабируем:
# - `Id`, потому что это не признак;
# - `Price`, потому что это целевая переменная;
# - бинарные признаки 0/1 (они уже в нужном масштабе).

# In[31]:


binary_cols = [c for c in df.columns if set(df[c].unique()) <= {0, 1}]
feature_cols = [c for c in df.columns if c not in ["Id", "Price", "DistrictId"] + binary_cols]
print("Бинарные (не масштабируем):", binary_cols)
print("Масштабируем:", feature_cols)


# In[32]:


std_scaler = StandardScaler()
df_std = df.copy()
df_std[feature_cols] = std_scaler.fit_transform(df[feature_cols])

mm_scaler = MinMaxScaler()
df_mm = df.copy()
df_mm[feature_cols] = mm_scaler.fit_transform(df[feature_cols])

show = ["Square", "Social_2", "HouseAge", "KitchenShare"]
pd.concat({
    "исходные": df[show].agg(["mean", "std", "min", "max"]),
    "StandardScaler": df_std[show].agg(["mean", "std", "min", "max"]),
    "MinMaxScaler": df_mm[show].agg(["mean", "std", "min", "max"]),
}).round(3)


# После `StandardScaler` у каждого признака среднее 0 и стандартное отклонение 1.
# После `MinMaxScaler` минимум равен 0, а максимум 1.
# Масштабирование **не меняет форму распределения**, меняется только шкала по оси X. Это хорошо видно на графике:

# In[33]:


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, data, title in zip(axes, [df, df_std, df_mm], ["Исходные", "StandardScaler", "MinMaxScaler"]):
    sns.histplot(data["Square"], bins=40, ax=ax, color=MAIN)
    ax.set_title(f"Square: {title}")
plt.tight_layout(); plt.show()


# In[34]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=df[feature_cols], ax=axes[0], color="#9aa5b4", showfliers=False)
axes[0].set_title("Признаки ДО масштабирования: несравнимые шкалы")
axes[0].tick_params(axis="x", rotation=70)
sns.boxplot(data=df_std[feature_cols], ax=axes[1], color=MAIN, showfliers=False)
axes[1].set_title("После StandardScaler: общий масштаб")
axes[1].tick_params(axis="x", rotation=70)
plt.tight_layout(); plt.show()


# > **Важно для будущих работ с моделями.** Scaler «учится» (`fit`) только на обучающей выборке,
# > а к тестовой применяется `transform` с уже найденными mean/std. Иначе в модель «утечёт»
# > информация из теста (data leakage). Здесь мы только готовим данные, поэтому `fit` делаем на всём датасете.
# 
# Сохраним подготовленные данные:

# In[35]:


df.to_csv("real_estate_clean.csv", index=False)
df_std.to_csv("real_estate_standardized.csv", index=False)
print("Итоговый размер:", df.shape)
df_std.head()


# ## Выводы
# 
# 1. Датасет из 10 000 квартир импортирован из zip-архива функцией `pd.read_csv`.
# 2. **Пропуски** найдены в `LifeSquare` (21%) и `Healthcare_1` (48%). Жилая площадь восстановлена через медианную долю
#    от общей, `Healthcare_1` заполнен медианой по району, а факт пропуска сохранён отдельным признаком.
# 3. **Аномалии** найдены методами IQR и Z-оценки, на гистограммах, BoxPlot и диаграммах рассеяния, а также логическими проверками:
#    год 20 052 011, 0 и 19 комнат, площадь 1 м², кухня 2 014 м², жилая площадь больше общей, дом с 0 этажей.
#    Ошибочные значения заменены медианами по группам, хвосты площадей ограничены перцентилями.
#    Графики «после» и повторные проверки подтверждают, что пропусков и аномалий не осталось.
# 4. **Сгенерированы признаки** `HouseAge`, `SquarePerRoom`, `KitchenShare`, `RelativeFloor`, `FloorCategory`, `DistrictSize`,
#    `Healthcare_1_missing`, `IsNew`.
#    **Кодирование:** бинаризация A/B → 0/1 (`Ecology_2`, `Ecology_3`, `Shops_2`), пороговая бинаризация `Binarizer` (`IsNew`),
#    One-Hot Encoding (`FloorCategory`), частотное кодирование (`DistrictId` → `DistrictSize`).
# 5. Выполнены **стандартизация** (`StandardScaler`: mean = 0, std = 1) и **нормализация** (`MinMaxScaler`: диапазон [0; 1]).
#    Данные готовы к обучению моделей.
