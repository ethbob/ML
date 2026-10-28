"""
ЛР1, часть 2. Функции Python и обработка исключений. Вариант 4.

Запуск:  python3 part2.py
"""

import random

VOWELS = set("аеёиоуыэюяaeiouy")


# ---------------------------------------------------------------------------
# Задание 1. Функция prime(n): n-е простое число (2 балла)
# ---------------------------------------------------------------------------
def is_prime(x):
    """True, если x простое. Делители достаточно искать до корня из x:
    если x = a * b и a > sqrt(x), то b < sqrt(x), и мы бы уже нашли b."""
    if x < 2:
        return False
    d = 2
    while d * d <= x:
        if x % d == 0:
            return False
        d += 1
    return True


def prime(n):
    """Возвращает простое число под номером n (нумерация с 1: prime(1) = 2)."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("номер должен быть натуральным числом")
    count = 0          # сколько простых уже нашли
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate


def task1():
    try:
        n = int(input("Номер простого числа: "))
        print(f"Простое число №{n}: {prime(n)}")
    except ValueError as e:
        print("Ошибка:", e)
    print("Первые 10 простых:", [prime(i) for i in range(1, 11)])


# ---------------------------------------------------------------------------
# Задание 2. Одна функция, разное поведение в зависимости от типа (4 балла)
# ---------------------------------------------------------------------------
def analyze(value):
    """Определяет тип аргумента через isinstance и выполняет нужную обработку."""

    # bool это подкласс int в Python (True == 1), поэтому отсекаем его отдельно
    if isinstance(value, bool):
        print("bool не обрабатывается")

    # ---- словарь: ключ с минимальным значением ----
    elif isinstance(value, dict):
        print("Тип: словарь", value)
        if not value:
            print("Словарь пуст")
            return
        # min проходит по ключам, key=value.get говорит сравнивать их по значениям
        key = min(value, key=value.get)
        print(f"Ключ с минимальным значением: {key!r} (значение {value[key]})")

    # ---- список: произведение между двумя первыми нулями + удаление повторов ----
    elif isinstance(value, list):
        print("Тип: список", value)
        zeros = [i for i, x in enumerate(value) if x == 0]   # индексы всех нулей
        if len(zeros) < 2:
            print("В списке меньше двух нулей, произведение посчитать нельзя")
        else:
            between = value[zeros[0] + 1: zeros[1]]           # срез строго между нулями
            if not between:
                print("Между первым и вторым нулём нет элементов")
            else:
                product = 1
                for x in between:
                    product *= x
                print(f"Элементы между первым и вторым нулём: {between}")
                print(f"Их произведение: {product}")
        # Удаляем повторы с сохранением порядка. set() тоже убирает повторы,
        # но порядок элементов в множестве не гарантирован.
        unique = list(dict.fromkeys(value))
        print("Список без повторов:", unique)

    # ---- число: все делители ----
    elif isinstance(value, int):
        print("Тип: целое число", value)
        if value == 0:
            print("У нуля бесконечно много делителей")
            return
        n = abs(value)
        divisors = [d for d in range(1, n + 1) if n % d == 0]
        print(f"Делители числа {value}: {divisors}")

    # ---- строка: палиндром? гласные и согласные ----
    elif isinstance(value, str):
        print("Тип: строка", repr(value))
        letters = [ch.lower() for ch in value if ch.isalpha()]   # только буквы, без пробелов
        clean = "".join(letters)
        if clean and clean == clean[::-1]:
            print("Строка является палиндромом (без учёта пробелов и регистра)")
        else:
            print("Строка не палиндром")
        vowels = sum(1 for ch in letters if ch in VOWELS)
        print(f"Гласных: {vowels}, согласных: {len(letters) - vowels}")
        # Мягкий и твёрдый знаки формально не согласные, но для простоты считаем их к ним.

    else:
        print(f"Тип {type(value).__name__} не поддерживается")


def task2():
    # «Сделать проверку со всеми этими случаями»: вызываем функцию для каждого типа
    tests = [
        {"яблоко": 5, "груша": 2, "слива": 9},
        [3, 0, 2, 4, 5, 0, 7, 3, 2],
        [1, 2, 3],
        36,
        "А роза упала на лапу Азора",
        "Машинное обучение",
        3.14,
    ]
    for t in tests:
        print("-" * 50)
        analyze(t)
    print("-" * 50)
    s = input("Своя строка для проверки (Enter: пропустить): ")
    if s:
        analyze(s)


# ---------------------------------------------------------------------------
# Задание 3. Сумма каждой строки матрицы m x n (2 балла)
# ---------------------------------------------------------------------------
def row_sums(matrix):
    """Матрица это список списков, каждая строка сама является списком."""
    return [sum(row) for row in matrix]


def task3():
    try:
        m = int(input("Число строк m: "))
        n = int(input("Число столбцов n: "))
    except ValueError:
        print("Нужны целые числа")
        return
    if m <= 0 or n <= 0:
        print("Размеры должны быть положительными")
        return

    mode = input("Заполнить случайно? (Enter: да, r: ввести вручную): ").strip().lower()
    if mode == "r":
        matrix = []
        for i in range(m):
            row = [int(x) for x in input(f"Строка {i + 1} ({n} чисел через пробел): ").split()]
            if len(row) != n:
                print(f"В строке должно быть ровно {n} чисел")
                return
            matrix.append(row)
    else:
        # Вложенный генератор: внешний делает m строк, внутренний по n чисел в строке
        matrix = [[random.randint(-9, 9) for _ in range(n)] for _ in range(m)]

    print("\nМатрица:")
    for row, s in zip(matrix, row_sums(matrix)):
        print(" ".join(f"{x:4}" for x in row), " | сумма =", s)


# ---------------------------------------------------------------------------
# Задание 4. Демонстрация try / except / finally (2 балла)
# ---------------------------------------------------------------------------
def safe_divide(a_text, b_text):
    try:
        # Код, в котором может возникнуть ошибка
        a = float(a_text)
        b = float(b_text)
        result = a / b
    except ValueError:
        # Сработает, если float() не смог превратить строку в число
        print(f"  except ValueError: '{a_text}' или '{b_text}' не число")
    except ZeroDivisionError:
        # Сработает при делении на ноль
        print("  except ZeroDivisionError: делить на ноль нельзя")
    else:
        # Выполняется, только если в try ошибок НЕ было
        print(f"  else: {a} / {b} = {result}")
    finally:
        # Выполняется ВСЕГДА: и при ошибке, и без неё.
        # Обычно здесь закрывают файлы и соединения.
        print("  finally: блок finally выполнен")


def task4():
    print("Три сценария: нормальный, деление на ноль, не число\n")
    for a, b in [("10", "4"), ("5", "0"), ("abc", "2")]:
        print(f"safe_divide('{a}', '{b}'):")
        safe_divide(a, b)
    print("\nТеперь сами:")
    safe_divide(input("Делимое: "), input("Делитель: "))


# ---------------------------------------------------------------------------
TASKS = {
    "1": ("Функция prime(n)", task1),
    "2": ("Функция, зависящая от типа аргумента", task2),
    "3": ("Суммы строк матрицы", task3),
    "4": ("try / except / finally", task4),
}


def main():
    while True:
        print("\n===== ЛР1, часть 2, вариант 4 =====")
        for key, (title, _) in TASKS.items():
            print(f"{key}. {title}")
        print("0. Выход")
        choice = input("Номер задания: ").strip()
        if choice == "0":
            break
        if choice in TASKS:
            print(f"\n--- Задание {choice}: {TASKS[choice][0]} ---")
            TASKS[choice][1]()
        else:
            print("Нет такого задания")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nВыход")
