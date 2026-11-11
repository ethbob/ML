"""
ЛР2. Разведочный анализ и подготовка данных (marketing_data).

Тот же код, что в lab2_marketing.ipynb, обычным скриптом.
Запуск (окружение активировано, из папки lab2_marketing):
    python lab2_marketing.py
Каждый график открывается отдельным окном: закрой окно, и выполнение продолжится.
"""


# # ЛР2. Разведочный анализ и подготовка данных
#
# **Датасет:** `data/marketing_data.zip` → `marketing_data.csv`. Это 2240 клиентов магазина: возраст, образование, семья, доход,
# траты по категориям товаров, каналы покупок, отклик на маркетинговые кампании.
#
# **Задание:**
# 1. импортировать датасет;
# 2. найти пропуски и аномальные значения (статистикой и графиками), заполнить пропуски, устранить выбросы и проверить результат графиками;
# 3. сгенерировать новые признаки (не меньше двух) и закодировать категориальные признаки (OHE, бинаризация и др.);
# 4. стандартизировать и нормализовать данные.
#
# | Столбец | Смысл |
# |---|---|
# | `ID` | номер клиента |
# | `Year_Birth` | год рождения |
# | `Education` | образование: Basic, Graduation (бакалавр), 2n Cycle / Master (магистр), PhD |
# | `Marital_Status` | семейное положение |
# | `Income` | годовой доход семьи, $ |
# | `Kidhome`, `Teenhome` | сколько в семье детей и подростков |
# | `Dt_Customer` | дата регистрации клиента |
# | `Recency` | сколько дней прошло с последней покупки |
# | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | траты за 2 года по категориям, $ |
# | `NumDealsPurchases` | покупки со скидкой |
# | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` | покупки через сайт, каталог, в магазине |
# | `NumWebVisitsMonth` | заходы на сайт за последний месяц |
# | `AcceptedCmp1`…`AcceptedCmp5`, `Response` | принял ли клиент предложение 1–5-й и последней кампании (0/1) |
# | `Complain` | жаловался ли за 2 года (0/1) |
# | `Country` | страна |


# ## 0. Проверка окружения
#
# Ноутбук запускается в виртуальном окружении `.venv`. Ячейка ниже показывает, **какой Python сейчас работает** и версии библиотек.
# Путь должен заканчиваться на `.venv/bin/python`: это значит, что среда активирована и ядро (kernel) использует её.

import sys
print("Python:", sys.version.split()[0])
print("Интерпретатор:", sys.executable)
print("Среда активирована:", ".venv" in sys.executable or sys.prefix != sys.base_prefix)


# ## 0.1 Библиотеки
#
# - **pandas** — таблицы (`DataFrame`): загрузка, фильтры, группировки;
# - **numpy** — математика над массивами;
# - **matplotlib / seaborn** — графики;
# - **scikit-learn** — кодирование и масштабирование признаков.

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, Binarizer)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

MAIN = "#3b6fb6"     # один цвет для всех «одиночных» графиков
GREY = "#9aa5b4"     # цвет для «до обработки»

def thousands(ax, axis="x"):
    # подписи оси в тысячах: 60000 -> 60k, чтобы подписи не налезали друг на друга
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)

for lib in (pd, np, matplotlib, sns, sklearn):
    print(f"{lib.__name__:12} {lib.__version__}")


# ## 1. Импорт датасета
#
# `pd.read_csv` читает CSV прямо из zip-архива, если в архиве один файл. Путь относительный: архив лежит в папке `data` рядом с ноутбуком.

df = pd.read_csv("data/marketing_data.zip")
print("Размер:", df.shape)
print(df.head())


# Названия столбцов проверяем сразу: у дохода имя записано с пробелами, `' Income '`. Из-за этого `df["Income"]` не сработал бы.
# `str.strip()` убирает пробелы по краям всех имён.

print(list(df.columns)[:6])
df.columns = df.columns.str.strip()
print(list(df.columns)[:6])

df.info()


# **Что видно в `info()`:**
# - `Income` и `Dt_Customer` хранятся как **строки** (`object` / `str`), а не как число и дата. С ними нельзя считать, сначала их нужно преобразовать;
# - у `Income` непустых значений меньше 2240, значит, есть пропуски.


# ### 1.1 Приведение типов
#
# **Доход.** Строка `"$84,835.00 "` содержит знак доллара, запятую-разделитель тысяч и пробел. Убираем лишние символы регулярным выражением
# `[$, ]` («любой из символов: доллар, запятая, пробел») и переводим в число. Пустые строки превращаются в `NaN` (пропуск).
#
# **Дата.** `pd.to_datetime` с форматом `%m/%d/%y` — месяц/день/год из двух цифр, например `6/16/14`.

raw_income = df["Income"].copy()
df["Income"] = pd.to_numeric(df["Income"].str.replace(r"[$, ]", "", regex=True), errors="coerce")
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%m/%d/%y")

print(pd.DataFrame({"было": raw_income.head(), "стало": df["Income"].head()}))

print("Тип Income:", df["Income"].dtype, "| тип Dt_Customer:", df["Dt_Customer"].dtype)
print("Даты регистрации: с", df["Dt_Customer"].min().date(), "по", df["Dt_Customer"].max().date())
raw = df.copy()      # копия до очистки: пригодится для графиков «до / после»
print(df.describe().T.round(1))


# `describe()` — первый инструмент поиска аномалий: смотрим на `min` и `max` и спрашиваем себя, бывает ли такое в жизни.
#
# - `Year_Birth` min = **1893**: в 2014 году (данные собраны в 2012–2014) клиенту было бы 121 год;
# - `Income` max = **666 666**: остальные доходы не выше ~163 000. Похоже на ввод «шестёрками» вместо реального числа;
# - `NumWebPurchases` до 27 и `NumCatalogPurchases` до 28 при медиане 4 и 2;
# - среднее у трат (`Mnt...`) заметно больше медианы (50%): распределения сильно скошены вправо.


# ## 2. Пропуски, дубликаты и аномальные значения
#
# ### 2.1 Дубликаты
#
# `ID` у всех клиентов разный, но если его не учитывать, некоторые строки полностью совпадают: тот же год рождения, доход,
# все траты и покупки до единицы. Скорее всего, это один клиент, записанный дважды.
# `duplicated()` помечает повторы (True для второго и следующих вхождений).

dup_mask = df.drop(columns="ID").duplicated()
print("Полных дубликатов (без учёта ID):", dup_mask.sum())
print(df[df.drop(columns="ID").duplicated(keep=False)].sort_values(["Year_Birth", "Income"]).head(4))


# ### 2.2 Пропуски
#
# `isna()` возвращает таблицу True/False (True — пропуск), `.sum()` считает пропуски по столбцам.

missing = df.isna().sum()
print(pd.DataFrame({"пропусков": missing, "%": (missing / len(df) * 100).round(2)})[missing > 0])

fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
sns.heatmap(df.isna().T, cbar=False, cmap=["#e8edf5", "#d1495b"], ax=axes[0], yticklabels=True)
axes[0].set_xticks([])
axes[0].set_title("Карта пропусков (красный — пропуск)")
axes[0].set_xlabel("строки датасета")
(missing[missing > 0]).plot.barh(ax=axes[1], color="#d1495b")
axes[1].set_title("Пропусков по столбцам")
plt.tight_layout(); plt.show()


# Пропуски есть только в `Income`: 24 значения (около 1%). Удалять эти строки не обязательно. Доход сильно зависит от образования,
# поэтому заполнять будем **медианой дохода по уровню образования**.


# ### 2.3 Аномалии: статистические методы
#
# **Метод межквартильного размаха (IQR).** Q1 и Q3 — 25-й и 75-й перцентили, IQR = Q3 − Q1.
# Выброс — значение меньше Q1 − 1.5·IQR или больше Q3 + 1.5·IQR. На этом правиле построены «усы» BoxPlot.
#
# **Метод Z-оценки.** z = (x − среднее) / стандартное отклонение, выброс при |z| > 3.
# Хорошо работает для распределений, похожих на нормальное. Само среднее и std сдвигаются от выбросов, поэтому IQR надёжнее.

num_cols = ["Year_Birth", "Income", "Recency", "MntWines", "MntMeatProducts", "MntGoldProds",
            "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
            "NumWebVisitsMonth"]

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

rows = []
for col in num_cols:
    s = df[col].dropna()
    lo, hi = iqr_bounds(s)
    z = (s - s.mean()) / s.std()
    rows.append({"признак": col,
                 "граница IQR снизу": round(lo, 1), "граница IQR сверху": round(hi, 1),
                 "выбросов по IQR": int(((s < lo) | (s > hi)).sum()),
                 "выбросов по |z|>3": int((z.abs() > 3).sum()),
                 "max": s.max()})
print(pd.DataFrame(rows))


# **Важно:** не всё, что IQR назвал выбросом, — ошибка. Траты (`Mnt...`) скошены вправо: богатых покупателей немного, но они настоящие,
# поэтому IQR находит у вина и мяса сотни «выбросов». Удалять их — значит выбросить лучших клиентов.
# Поэтому статистику дополняем **логическими проверками**, то есть здравым смыслом про предметную область:

mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
REF_YEAR = df["Dt_Customer"].max().year          # 2014 — год последней регистрации

def logical_checks(d):
    total_purch = d[purchase_cols].sum(axis=1)
    total_spent = d[mnt_cols].sum(axis=1)
    return pd.Series({
        "дубликаты строк (без ID)": int(d.drop(columns="ID").duplicated().sum()),
        "пропуски Income": int(d["Income"].isna().sum()),
        f"возраст > 90 лет (Year_Birth < {REF_YEAR - 90})": int((d["Year_Birth"] < REF_YEAR - 90).sum()),
        "Income > 200 000": int((d["Income"] > 200_000).sum()),
        "Marital_Status: YOLO / Absurd / Alone": int(d["Marital_Status"].isin(["YOLO", "Absurd", "Alone"]).sum()),
        "покупок 0, а траты > 0": int(((total_purch == 0) & (total_spent > 0)).sum()),
        "NumWebPurchases > 20": int((d["NumWebPurchases"] > 20).sum()),
        "NumCatalogPurchases > 20": int((d["NumCatalogPurchases"] > 20).sum()),
    }, name="строк")

print(logical_checks(df).to_frame())

print(df["Marital_Status"].value_counts().to_string())
print()
print(df["Education"].value_counts().to_string())


# **Категориальные аномалии:**
# - `YOLO` («живём один раз») и `Absurd` — явный мусор при заполнении анкеты, а не семейное положение;
# - `Alone` («один») — по смыслу то же, что `Single`;
# - в `Education` значение `2n Cycle` — это европейское название магистратуры («второй цикл» Болонской системы), то есть то же, что `Master`.


# ### 2.4 Аномалии: графики
#
# **Гистограммы** показывают форму распределения: длинный хвост или одиночный столбик далеко в стороне выдают выбросы.

cols = ["Year_Birth", "Income", "MntWines", "MntMeatProducts", "NumWebPurchases", "NumCatalogPurchases"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), cols):
    sns.histplot(df[col].dropna(), bins=50, ax=ax, color=MAIN)
    ax.set_title(col)
    if col == "Income":
        thousands(ax)
plt.suptitle("Распределения ДО очистки", fontsize=14)
plt.tight_layout(); plt.show()


# **BoxPlot (ящик с усами):** ящик — от Q1 до Q3, линия внутри — медиана, усы — до 1.5·IQR, точки за усами — кандидаты в выбросы.
# У `Year_Birth` три точки далеко внизу (1893–1900), у `Income` одна точка 666 666 сплющивает весь ящик.

cols = ["Year_Birth", "Income", "NumWebPurchases", "NumCatalogPurchases", "NumWebVisitsMonth"]
fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
for ax, col in zip(axes, cols):
    sns.boxplot(y=df[col], ax=ax, color=MAIN, flierprops={"markersize": 3})
    ax.set_title(col); ax.set_ylabel("")
plt.suptitle("BoxPlot ДО очистки", fontsize=14)
plt.tight_layout(); plt.show()


# **Диаграмма рассеяния** «доход → траты»: в норме чем больше доход, тем больше тратит клиент.
# Точка с доходом 666 666 и обычными тратами выбивается из общей картины.

total_spent_raw = df[mnt_cols].sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Income"], total_spent_raw, s=8, alpha=0.5, color=MAIN)
ax.set_xlabel("Income, $"); ax.set_ylabel("Траты за 2 года, $")
thousands(ax)
ax.set_title("Доход и траты: точка справа — аномалия 666 666")
plt.tight_layout(); plt.show()


# ### 2.5 Устранение проблем и заполнение пропусков
#
# Принцип: **ошибочное значение → превращаем в пропуск → заполняем разумным значением**. Строки удаляем, только если в них ошибка,
# которую нельзя исправить, и таких строк очень мало.
# Заполняем **медианой**: в отличие от среднего, она не сдвигается от выбросов. И по возможности **медианой по группе**, а не по всем данным.
#
# **1) Дубликаты удаляем.** Повтор одного и того же клиента искажает статистику: его траты считались бы дважды.

n_before = len(df)
df = df.loc[~df.drop(columns="ID").duplicated()].reset_index(drop=True)
print(f"Было {n_before} строк, удалено {n_before - len(df)}, осталось {len(df)}")


# **2) Категории.** Приводим к осмысленному набору значений: `Alone` → `Single`; `YOLO`/`Absurd` → пропуск, затем заполняем **модой**
# (самым частым значением). Для категорий медиана не имеет смысла, поэтому используется мода. `2n Cycle` → `Master`.

df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single", "YOLO": np.nan, "Absurd": np.nan})
mode_status = df["Marital_Status"].mode()[0]
df["Marital_Status"] = df["Marital_Status"].fillna(mode_status)
df["Education"] = df["Education"].replace({"2n Cycle": "Master"})
print("Мода Marital_Status:", mode_status)
print(df["Marital_Status"].value_counts().to_dict())
print(df["Education"].value_counts().to_dict())


# **3) Год рождения.** Значения раньше 1924 (возраст больше 90) — ошибка ввода. Заменяем медианой.

df.loc[df["Year_Birth"] < REF_YEAR - 90, "Year_Birth"] = np.nan
df["Year_Birth"] = df["Year_Birth"].fillna(df["Year_Birth"].median()).astype(int)
print("Год рождения теперь от", df["Year_Birth"].min(), "до", df["Year_Birth"].max())


# **4) Доход.** Значение 666 666 превращаем в пропуск, а затем заполняем все пропуски (исходные 24 и этот) **медианой дохода по образованию**.
# `groupby("Education")["Income"].transform("median")` для каждой строки возвращает медиану её группы.

df.loc[df["Income"] > 200_000, "Income"] = np.nan
print("Медиана дохода по образованию:")
print(df.groupby("Education")["Income"].median().round(0).to_string())
df["Income"] = df["Income"].fillna(df.groupby("Education")["Income"].transform("median"))
print("\nПропусков в Income осталось:", df["Income"].isna().sum())


# **5) «Покупок 0, а траты есть».** Клиент не может потратить деньги, ни разу ничего не купив. Восстановить правильное число покупок
# не из чего, а таких строк всего несколько (меньше 0.3%), поэтому эти строки удаляем.

total_purch = df[purchase_cols].sum(axis=1)
bad = (total_purch == 0) & (df[mnt_cols].sum(axis=1) > 0)
print("Удаляем строк:", bad.sum())
df = df.loc[~bad].reset_index(drop=True)


# **6) Ограничение хвостов (винзоризация).** Значения 23–28 покупок через сайт или каталог при медиане 2–4 — редкие единичные
# экстремумы. Не удаляем их, а «прижимаем» к 99.5-му перцентилю методом `clip`.
# Траты (`Mnt...`) **не трогаем**: большие траты реальны, это просто самые активные клиенты.

for col in ["NumWebPurchases", "NumCatalogPurchases", "NumWebVisitsMonth"]:
    hi = df[col].quantile(0.995)
    n_cut = (df[col] > hi).sum()
    df[col] = df[col].clip(upper=hi)
    print(f"{col}: верхняя граница {hi:.0f}, ограничено значений: {n_cut}")


# ### 2.6 Проверка: пропусков и аномалий не осталось

print(pd.DataFrame({"до очистки": logical_checks(raw), "после очистки": logical_checks(df)}))

print("Всего пропусков в данных:", int(df.isna().sum().sum()))
fig, ax = plt.subplots(figsize=(14, 3.5))
sns.heatmap(df.isna().T, cbar=False, cmap=["#e8edf5", "#d1495b"], ax=ax, yticklabels=True, vmin=0, vmax=1)
ax.set_xticks([])
ax.set_title("Карта пропусков ПОСЛЕ обработки: красных штрихов нет")
plt.tight_layout(); plt.show()


# **BoxPlot до и после.** Верхний ряд — исходные данные (серый), нижний — очищенные (синий). Каждый признак со своей шкалой.

cols = ["Year_Birth", "Income", "NumWebPurchases", "NumCatalogPurchases", "NumWebVisitsMonth"]
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
for i, col in enumerate(cols):
    sns.boxplot(y=raw[col], ax=axes[0, i], color=GREY, flierprops={"markersize": 3})
    sns.boxplot(y=df[col], ax=axes[1, i], color=MAIN, flierprops={"markersize": 3})
    axes[0, i].set_title(col); axes[0, i].set_ylabel(""); axes[1, i].set_ylabel("")
axes[0, 0].set_ylabel("ДО", fontsize=13); axes[1, 0].set_ylabel("ПОСЛЕ", fontsize=13)
plt.suptitle("BoxPlot до и после очистки", fontsize=14)
plt.tight_layout(); plt.show()


# Оставшиеся точки за усами — это реальные данные (клиенты 1940-х годов рождения, доходы 100–160 тыс.), а не ошибки.
# Исчезли невозможные значения, из-за которых на верхнем ряду ящики сжимались в полоску.

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Income"], df[mnt_cols].sum(axis=1), s=8, alpha=0.5, color=MAIN)
ax.set_xlabel("Income, $"); ax.set_ylabel("Траты за 2 года, $")
thousands(ax)
ax.set_title("ПОСЛЕ: аномальный доход исчез, видна связь «больше доход — больше трат»")
plt.tight_layout(); plt.show()


# ### 2.7 Корреляции
#
# Коэффициент корреляции Пирсона — от −1 до 1: около 0 — линейной связи нет, ближе к ±1 — сильная связь.

corr_cols = ["Income", "Year_Birth", "Kidhome", "Teenhome", "Recency"] + mnt_cols + \
            ["NumDealsPurchases"] + purchase_cols + ["NumWebVisitsMonth", "Response"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, square=True, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Матрица корреляций (после очистки)")
plt.tight_layout(); plt.show()
print(corr["Income"].drop("Income").sort_values(ascending=False).round(2).head(6))


# Доход сильно связан с тратами на вино и мясо и с покупками по каталогу. Дети (`Kidhome`) связаны с тратами отрицательно:
# семьи с маленькими детьми тратят меньше. А чем больше заходов на сайт, тем ниже доход: это охотники за скидками.


# ## 3. Новые признаки и кодирование категориальных
#
# ### 3.1 Генерация новых признаков (feature engineering)
#
# | Признак | Формула | Зачем |
# |---|---|---|
# | `Age` | 2014 − Year_Birth | возраст понятнее года рождения |
# | `Customer_Days` | дней от регистрации до последней даты в данных | как давно клиент с магазином |
# | `Children` | Kidhome + Teenhome | общее число детей |
# | `Is_Parent` | 1, если Children > 0 | есть ли дети вообще |
# | `Family_Size` | взрослые (1 или 2) + дети | размер семьи |
# | `Total_Spent` | сумма всех `Mnt...` | сколько клиент потратил всего |
# | `Total_Purchases` | сайт + каталог + магазин | сколько раз покупал |
# | `Avg_Check` | Total_Spent / Total_Purchases | средний чек |
# | `Total_Accepted` | сумма `AcceptedCmp1..5` | сколько кампаний клиент принял |

last_date = df["Dt_Customer"].max()
df["Age"] = REF_YEAR - df["Year_Birth"]
df["Customer_Days"] = (last_date - df["Dt_Customer"]).dt.days
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["Is_Parent"] = (df["Children"] > 0).astype(int)

partner = df["Marital_Status"].isin(["Married", "Together"])
df["Family_Size"] = np.where(partner, 2, 1) + df["Children"]

df["Total_Spent"] = df[mnt_cols].sum(axis=1)
df["Total_Purchases"] = df[purchase_cols].sum(axis=1)
df["Avg_Check"] = (df["Total_Spent"] / df["Total_Purchases"]).round(2)
df["Total_Accepted"] = df[[f"AcceptedCmp{i}" for i in range(1, 6)]].sum(axis=1)

new_cols = ["Age", "Customer_Days", "Children", "Is_Parent", "Family_Size",
            "Total_Spent", "Total_Purchases", "Avg_Check", "Total_Accepted"]
print(df[new_cols].describe().T.round(1))


# `np.where(условие, a, b)` — это «если / иначе» для целого столбца: где условие истинно, берётся `a`, иначе `b`.
# `.dt.days` достаёт из разницы дат число дней.
#
# Проверим, что новые признаки действительно что-то говорят о клиентах:

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.boxplot(data=df, x="Is_Parent", y="Total_Spent", ax=axes[0], color=MAIN)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["без детей", "есть дети"])
axes[0].set_xlabel(""); axes[0].set_title("Траты: родители и не родители")

axes[1].scatter(df["Income"], df["Total_Spent"], s=8, alpha=0.4, color=MAIN)
axes[1].set_xlabel("Income, $"); axes[1].set_ylabel("Total_Spent, $"); thousands(axes[1])
axes[1].set_title("Доход и общие траты")

resp = df.groupby("Total_Accepted")["Response"].mean()
axes[2].bar(resp.index.astype(str), resp.values, color=MAIN)
axes[2].set_xlabel("сколько прошлых кампаний принял"); axes[2].set_ylabel("доля откликнувшихся")
axes[2].set_title("Отклик на последнюю кампанию")
plt.tight_layout(); plt.show()

print(df.groupby("Is_Parent")["Total_Spent"].median().rename({0: "без детей", 1: "есть дети"}))


# Клиенты без детей тратят в разы больше, чем родители, а клиенты, принимавшие прошлые кампании, гораздо чаще откликаются на новую.
# Значит, признаки `Is_Parent` и `Total_Accepted` полезны для модели.


# ### 3.2 Кодирование категориальных признаков
#
# Модели машинного обучения работают только с числами, поэтому текстовые категории нужно перевести в числа.
# У нас три категориальных признака, и для каждого подходит свой способ:
#
# | Признак | Есть ли порядок | Способ |
# |---|---|---|
# | `Education` | да: Basic < Graduation < Master < PhD | **порядковое кодирование** (OrdinalEncoder) |
# | `Marital_Status` | нет, но по смыслу две группы: с партнёром / без | **бинаризация** → `Living_With_Partner` |
# | `Country` | нет, 8 стран | **One-Hot Encoding** |
#
# **Порядковое (ordinal) кодирование.** У образования есть естественный порядок, поэтому числа 0, 1, 2, 3 здесь осмысленны:
# PhD действительно «выше» Basic. Порядок категорий задаём явно, иначе энкодер расставит их по алфавиту.

edu_order = [["Basic", "Graduation", "Master", "PhD"]]
ord_enc = OrdinalEncoder(categories=edu_order)
df["Education_Level"] = ord_enc.fit_transform(df[["Education"]]).astype(int)
print(df[["Education", "Education_Level"]].drop_duplicates().sort_values("Education_Level"))


# **Бинаризация.** `Marital_Status` сводим к двум значениям: живёт с партнёром (`Married`, `Together`) → 1, один (`Single`, `Divorced`, `Widow`) → 0.

df["Living_With_Partner"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)
print(pd.crosstab(df["Marital_Status"], df["Living_With_Partner"]))


# **Бинаризация числового признака по порогу** (`Binarizer`): значение выше порога → 1, иначе → 0.
# Признак `High_Income`: доход выше медианы.

income_median = df["Income"].median()
binarizer = Binarizer(threshold=income_median)
df["High_Income"] = binarizer.fit_transform(df[["Income"]]).astype(int).ravel()
print(f"Порог (медиана дохода): {income_median:,.0f} $")
print(df["High_Income"].value_counts())


# **One-Hot Encoding (OHE).** Признак с k категориями превращается в k столбцов из 0 и 1, в каждой строке ровно одна единица.
# Просто пронумеровать страны нельзя: модель решила бы, что США (7) «больше» Испании (5), хотя порядка у стран нет.
#
# Ниже показаны два способа: `pd.get_dummies` (pandas) и `OneHotEncoder` (sklearn).

# Способ 1: pandas
country_dummies = pd.get_dummies(df["Country"], prefix="Country", dtype=int)
print(country_dummies.head())

# Способ 2: sklearn — энкодер запоминает категории (fit), а потом одинаково кодирует новые данные (transform)
ohe = OneHotEncoder(handle_unknown="ignore")
encoded = ohe.fit_transform(df[["Country"]]).toarray().astype(int)
print("\nКатегории, найденные OneHotEncoder:", list(ohe.categories_[0]))
print("Результаты двух способов совпадают:", (encoded == country_dummies.values).all())


# Добавляем закодированные столбцы в таблицу и убираем исходные текстовые признаки и дату: в закодированном виде они уже есть.

df_enc = pd.concat([df, country_dummies], axis=1).drop(columns=["Education", "Marital_Status", "Country", "Dt_Customer"])
print("Итоговый размер:", df_enc.shape)
print("Нечисловых столбцов осталось:", list(df_enc.select_dtypes(exclude="number").columns))
print(df_enc.head())


# ## 4. Стандартизация и нормализация
#
# Признаки в разных масштабах: `Income` — десятки тысяч, `Kidhome` — 0…2, `Avg_Check` — десятки.
# Алгоритмы, которые считают расстояния (kNN, k-means, SVM) или используют градиентный спуск (линейные модели, нейросети),
# начинают ориентироваться на признак с большими числами просто из-за его масштаба.
#
# | Метод | Формула | Результат | Когда |
# |---|---|---|---|
# | **Стандартизация** (`StandardScaler`) | z = (x − mean) / std | среднее 0, стандартное отклонение 1 | по умолчанию: линейные модели, SVM, PCA |
# | **Нормализация** (`MinMaxScaler`) | x' = (x − min) / (max − min) | все значения в [0; 1] | нужен фиксированный диапазон: нейросети, kNN |
#
# **Не масштабируем:** `ID` (это не признак), `Response` (целевая переменная), бинарные признаки 0/1 — они уже в нужном масштабе.

binary_cols = [c for c in df_enc.columns if set(df_enc[c].unique()) <= {0, 1}]
scale_cols = [c for c in df_enc.columns if c not in ["ID", "Response"] + binary_cols]
print("Бинарные (не масштабируем):", binary_cols)
print("\nМасштабируем:", scale_cols)

std_scaler = StandardScaler()
df_std = df_enc.copy()
df_std[scale_cols] = std_scaler.fit_transform(df_enc[scale_cols])

mm_scaler = MinMaxScaler()
df_mm = df_enc.copy()
df_mm[scale_cols] = mm_scaler.fit_transform(df_enc[scale_cols])

show = ["Income", "Age", "Total_Spent", "Kidhome"]
print(pd.concat({
    "исходные": df_enc[show].agg(["mean", "std", "min", "max"]),
    "StandardScaler": df_std[show].agg(["mean", "std", "min", "max"]),
    "MinMaxScaler": df_mm[show].agg(["mean", "std", "min", "max"]),
}).round(3))


# После `StandardScaler` у каждого признака среднее 0 и std 1, после `MinMaxScaler` — min 0 и max 1.
# **Форма распределения при масштабировании не меняется**, меняется только шкала по оси X:

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, data, title in zip(axes, [df_enc, df_std, df_mm], ["Исходные", "StandardScaler", "MinMaxScaler"]):
    sns.histplot(data["Income"], bins=40, ax=ax, color=MAIN)
    ax.set_title(f"Income: {title}")
    if title == "Исходные":
        thousands(ax)
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(data=df_enc[scale_cols], ax=axes[0], color=GREY, showfliers=False)
axes[0].set_title("ДО масштабирования: Income и Total_Spent «задавили» остальные")
axes[0].tick_params(axis="x", rotation=80)
sns.boxplot(data=df_std[scale_cols], ax=axes[1], color=MAIN, showfliers=False)
axes[1].set_title("После StandardScaler: общий масштаб")
axes[1].tick_params(axis="x", rotation=80)
plt.tight_layout(); plt.show()


# > **Важно для будущих работ с моделями.** Scaler «учится» (`fit`) только на обучающей выборке, а к тестовой применяется `transform`
# > с уже найденными mean и std. Иначе в модель «утечёт» информация из теста (data leakage). Здесь моделей нет, только подготовка данных,
# > поэтому `fit` делаем на всём датасете.
#
# Сохраняем результаты (файлы появятся в папке `output` рядом с ноутбуком):

import os
os.makedirs("output", exist_ok=True)
df_enc.to_csv("output/marketing_clean.csv", index=False)
df_std.to_csv("output/marketing_standardized.csv", index=False)
df_mm.to_csv("output/marketing_normalized.csv", index=False)
print("Сохранено:", os.listdir("output"))
print("Итоговый датасет:", df_std.shape)


# ## Выводы
#
# 1. **Импорт.** Датасет из 2240 клиентов загружен из zip-архива функцией `pd.read_csv`. Имена столбцов очищены от пробелов,
#    доход из строки `"$84,835.00"` переведён в число, дата регистрации — в тип `datetime`.
# 2. **Пропуски и аномалии** найдены методами IQR и Z-оценки, на гистограммах, BoxPlot и диаграмме рассеяния, а также логическими проверками:
#    47 полных дубликатов, 24 пропуска дохода, доход 666 666, годы рождения 1893–1900, бессмысленные категории `YOLO` / `Absurd`,
#    клиенты с тратами, но без покупок. Дубликаты и противоречивые строки удалены, ошибочные значения заменены медианами
#    (доход — по группе образования), категории приведены к единому виду, редкие экстремумы ограничены 99.5-м перцентилем.
#    Повторные проверки и графики «после» подтверждают, что пропусков и аномалий не осталось.
# 3. **Новые признаки:** `Age`, `Customer_Days`, `Children`, `Is_Parent`, `Family_Size`, `Total_Spent`, `Total_Purchases`, `Avg_Check`,
#    `Total_Accepted`, а также `Living_With_Partner` и `High_Income`. **Кодирование:** порядковое (`Education` → `Education_Level`),
#    бинаризация (`Marital_Status` → `Living_With_Partner`), пороговая бинаризация `Binarizer` (`High_Income`),
#    One-Hot Encoding (`Country`, двумя способами).
# 4. **Масштабирование:** стандартизация `StandardScaler` (mean = 0, std = 1) и нормализация `MinMaxScaler` (диапазон [0; 1]).
#    Данные готовы к обучению моделей.
