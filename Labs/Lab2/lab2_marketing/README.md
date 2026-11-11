# ММО, ЛР2: разведочный анализ и подготовка данных (marketing_data)

| Файл / папка | Что это |
|---|---|
| `lab2_marketing.ipynb` | работа целиком: код, графики, объяснения (уже выполнена, всё видно сразу) |
| `lab2_marketing.py` | тот же код обычным скриптом: `python lab2_marketing.py` (графики открываются окнами — закрыл окно, выполнение идёт дальше) |
| `data/marketing_data.zip` | датасет: 2240 клиентов магазина, 28 признаков |
| `requirements.txt` | библиотеки для окружения |
| `output/` | появится после запуска: очищенные и масштабированные данные (CSV) |

Ноутбук проверен на **Python 3.8** и **Python 3.13** (pandas 2.0 и 3.0).

---

## Запуск на Mac в PyCharm

### 1. Создать окружение и поставить библиотеки (один раз)

Открой **Terminal** (или вкладку Terminal внизу в PyCharm):

```bash
cd ~/Documents/ML/Labs/Lab2/lab2_marketing
python3 -m venv .venv                 # создаём окружение в папке .venv
source .venv/bin/activate             # активируем: в начале строки появится (.venv)
pip install --upgrade pip
pip install -r requirements.txt
```

Проверка, что окружение активно:
```bash
which python        # должно быть .../lab2_marketing/.venv/bin/python
```

> У тебя в старом `lab2_eda/.venv` стоит Python 3.8 — он давно не поддерживается.
> Если хочешь новее: `brew install python@3.12` и создавай окружение командой
> `python3.12 -m venv .venv`. Ноутбук работает на обеих версиях.

### 2. Открыть в PyCharm

1. **File → Open…** → выбрать папку `lab2_marketing` → **Open**.
2. Подключить окружение: **PyCharm → Settings** (⌘,) → **Project: lab2_marketing → Python Interpreter**
   → **Add Interpreter → Add Local Interpreter → Existing** → указать
   `~/Documents/ML/Labs/Lab2/lab2_marketing/.venv/bin/python` → **OK**.
   Справа внизу в строке состояния должно появиться `Python 3.x (lab2_marketing)` / `.venv`.
3. Открыть `lab2_marketing.ipynb`.
4. Сверху в панели ноутбука убедиться, что выбран **Jupyter Server: Managed** и ядро **Python 3 (.venv)**.
   Если PyCharm предложит установить `jupyter` / `ipykernel` — согласиться (поставит в это же окружение).
5. **Run All** (двойная зелёная стрелка ▶▶ над ноутбуком, или ⌃⇧⏎ / Shift+Ctrl+Enter).

После Run All весь код **загружен в память ядра**: переменные `df`, `df_std`, `df_mm` доступны, любую ячейку
можно перезапустить или дописать новую на защите (например, `df.describe()`).

Первая ячейка ноутбука печатает путь интерпретатора и `Среда активирована: True` — это и есть доказательство,
что ноутбук работает в окружении `.venv`.

> Работа с `.ipynb` есть в PyCharm Professional и в новом едином PyCharm (2025+).
> Если в твоей версии ноутбуки не открываются — запусти из терминала (окружение активировано):
> `jupyter notebook lab2_marketing.ipynb` → откроется в браузере → **Kernel → Restart & Run All**.

### Частые проблемы

| Проблема | Решение |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | в PyCharm выбран не тот интерпретатор (см. шаг 2) или не выполнен `pip install -r requirements.txt` |
| `FileNotFoundError: data/marketing_data.zip` | ноутбук открыт не из папки `lab2_marketing`; рабочая папка ядра должна быть папкой ноутбука |
| В первой ячейке `Среда активирована: False` | ядро запущено на системном Python — выбрать `.venv` в списке ядер сверху |
| Графики не показываются | перезапустить ядро: **Restart Kernel** и снова **Run All** |

---

## Что сделано (по пунктам задания)

1. **Импорт:** `pd.read_csv` прямо из zip; имена столбцов очищены от пробелов; доход `"$84,835.00"` → число;
   дата `6/16/14` → `datetime`.
2. **Пропуски и аномалии**
   - поиск: `info`, `describe`, `isna`, `duplicated`, IQR, Z-оценка, логические проверки;
   - графики: карта пропусков, гистограммы, BoxPlot, диаграмма рассеяния, матрица корреляций;
   - найдено: 47 дубликатов, 24 пропуска дохода, доход 666 666, годы рождения 1893–1900,
     категории `YOLO` / `Absurd` / `Alone`, `2n Cycle` (= Master), 6 клиентов с тратами без покупок,
     единичные 23–28 покупок через сайт/каталог;
   - исправление: удаление дубликатов и противоречивых строк, медиана (доход — по образованию),
     мода для категорий, винзоризация хвостов;
   - проверка: те же проверки и графики после — всё по нулям.
3. **Новые признаки:** `Age`, `Customer_Days`, `Children`, `Is_Parent`, `Family_Size`, `Total_Spent`,
   `Total_Purchases`, `Avg_Check`, `Total_Accepted`, `Living_With_Partner`, `High_Income`.
   **Кодирование:** порядковое (`Education` → `OrdinalEncoder`), бинаризация (`Marital_Status`),
   `Binarizer` по порогу (`High_Income`), One-Hot (`Country`: `get_dummies` и `OneHotEncoder`).
4. **Масштабирование:** `StandardScaler` (mean 0, std 1) и `MinMaxScaler` ([0; 1]), сравнение до/после.
