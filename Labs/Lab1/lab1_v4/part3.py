"""
ЛР1, часть 3. Классы и ООП. Вариант 4.

Запуск:  python3 part3.py
"""

import math


# ===========================================================================
# Задание 1. Класс Square (1 балл)
# ===========================================================================
class Square:
    def __init__(self, side):
        # __init__ это конструктор: вызывается при создании объекта Square(5).
        # self это ссылка на сам создаваемый объект, через неё хранятся его данные.
        if side <= 0:
            raise ValueError("сторона должна быть положительной")
        self.side = side

    def perimeter(self):
        return 4 * self.side

    def area(self):
        return self.side ** 2

    def diagonal(self):
        # По теореме Пифагора: d^2 = a^2 + a^2, значит d = a * sqrt(2)
        return self.side * math.sqrt(2)


def task1():
    try:
        a = float(input("Сторона квадрата: "))
        sq = Square(a)
    except ValueError as e:
        print("Ошибка:", e)
        return
    print(f"Периметр:  {sq.perimeter():.2f}")
    print(f"Площадь:   {sq.area():.2f}")
    print(f"Диагональ: {sq.diagonal():.2f}")


# ===========================================================================
# Задание 2. Наследование и переопределение: Stationery (3 балла)
# ===========================================================================
class Stationery:
    """Родительский (базовый) класс: канцелярская принадлежность."""

    def __init__(self, title):
        self.title = title

    def draw(self):
        print(f"[{self.title}] Запуск отрисовки")


class Pen(Stationery):
    # Pen(Stationery) означает: Pen наследует всё от Stationery (атрибут title,
    # конструктор, метод draw). Ниже мы ПЕРЕОПРЕДЕЛЯЕМ draw своей версией.
    def draw(self):
        print(f"[{self.title}] Ручка: рисуем ровную синюю линию")


class Pencil(Stationery):
    def draw(self):
        print(f"[{self.title}] Карандаш: делаем набросок, его можно стереть ластиком")


class Handle(Stationery):
    def draw(self):
        # super() даёт доступ к методу родителя: сначала общее сообщение, потом своё
        super().draw()
        print(f"[{self.title}] Маркер: выделяем текст жирной жёлтой полосой")


def task2():
    items = [
        Stationery("Принадлежность"),
        Pen("Ручка Parker"),
        Pencil("Карандаш Koh-i-Noor"),
        Handle("Маркер Stabilo"),
    ]
    # Полиморфизм: у всех объектов вызываем один и тот же метод draw(),
    # а Python сам выбирает версию метода по классу объекта.
    for item in items:
        print(f"{type(item).__name__}.draw():")
        item.draw()
        print()


# ===========================================================================
# Задание 3. Классы Faculty и Student
# ===========================================================================
class Faculty:
    def __init__(self, name):
        self.__name = name   # два подчёркивания: «закрытое» поле, доступ через методы

    def get_name(self):
        return self.__name

    def set_name(self, name):
        if not name.strip():
            raise ValueError("название факультета не может быть пустым")
        self.__name = name

    def __str__(self):
        # __str__ определяет, как объект выглядит в print()
        return self.__name


class Student:
    PASS_MARK = 4        # атрибут класса: общий для всех студентов (10-балльная шкала)

    def __init__(self, full_name, birth_year, faculty, grades):
        # Поля закрытые. Значения записываем через set-методы, чтобы сразу
        # сработала проверка корректности.
        self.set_full_name(full_name)
        self.set_birth_year(birth_year)
        self.set_faculty(faculty)
        self.set_grades(grades)

    # ---- методы доступа (геттеры) и изменения (сеттеры) ----
    def get_full_name(self):
        return self.__full_name

    def set_full_name(self, full_name):
        if not full_name.strip():
            raise ValueError("ФИО не может быть пустым")
        self.__full_name = full_name.strip()

    def get_birth_year(self):
        return self.__birth_year

    def set_birth_year(self, year):
        if not 1900 < year < 2020:
            raise ValueError("некорректный год рождения")
        self.__birth_year = year

    def get_faculty(self):
        return self.__faculty

    def set_faculty(self, faculty):
        if not isinstance(faculty, Faculty):
            raise TypeError("факультет должен быть объектом Faculty")
        self.__faculty = faculty

    def get_grades(self):
        return dict(self.__grades)        # отдаём копию, чтобы словарь не испортили снаружи

    def set_grades(self, grades):
        for subject, mark in grades.items():
            if not 0 <= mark <= 10:
                raise ValueError(f"оценка по «{subject}» должна быть от 0 до 10")
        self.__grades = dict(grades)

    def set_grade(self, subject, mark):
        """Изменить одну оценку (например, после пересдачи)."""
        grades = self.get_grades()
        grades[subject] = mark
        self.set_grades(grades)

    # ---- успеваемость ----
    def average(self):
        return sum(self.__grades.values()) / len(self.__grades) if self.__grades else 0

    def debts(self):
        return [s for s, m in self.__grades.items() if m < self.PASS_MARK]

    def status(self):
        marks = self.__grades.values()
        if self.debts():
            return "есть задолженности"
        if all(m >= 9 for m in marks):
            return "отличник"
        if all(m >= 6 for m in marks):
            return "хорошист"
        return "успевает"

    def report(self):
        lines = [
            f"{self.__full_name}, {self.__birth_year} г. р., факультет {self.__faculty}",
            "  Оценки: " + ", ".join(f"{s} – {m}" for s, m in self.__grades.items()),
            f"  Средний балл: {self.average():.2f}; статус: {self.status()}",
        ]
        if self.debts():
            lines.append("  Долги по: " + ", ".join(self.debts()))
        return "\n".join(lines)


def read_students():
    """Считывает студентов с клавиатуры. Enter вместо числа: демонстрационные данные."""
    raw = input("Сколько студентов ввести? (Enter: демо-данные): ").strip()
    if not raw:
        fit = Faculty("ФИТиУ")
        fksis = Faculty("ФКСиС")
        return [
            Student("Иванов Иван Иванович", 2004, fit,
                    {"Математика": 9, "Программирование": 10, "Физика": 9}),
            Student("Петрова Анна Сергеевна", 2005, fksis,
                    {"Математика": 7, "Программирование": 8, "Физика": 6}),
            Student("Сидоров Павел Олегович", 2004, fit,
                    {"Математика": 3, "Программирование": 6, "Физика": 5}),
        ]

    faculties = {}              # один объект Faculty на каждое название
    students = []
    for i in range(int(raw)):
        print(f"\nСтудент {i + 1}")
        name = input("  ФИО: ")
        year = int(input("  Год рождения: "))
        fac_name = input("  Факультет: ").strip()
        faculty = faculties.setdefault(fac_name, Faculty(fac_name))
        grades = {}
        print("  Оценки в формате «предмет оценка», пустая строка: конец")
        while True:
            line = input("    ").strip()
            if not line:
                break
            subject, mark = line.rsplit(maxsplit=1)   # делим по последнему пробелу
            grades[subject] = int(mark)
        students.append(Student(name, year, faculty, grades))
    return students


def task3():
    try:
        students = read_students()
    except (ValueError, TypeError) as e:
        print("Ошибка ввода:", e)
        return

    print("\n=== Успеваемость ===")
    for st in students:
        print(st.report())

    # Демонстрация методов изменения: студент пересдал экзамен
    debtor = next((st for st in students if st.debts()), None)
    if debtor:
        subject = debtor.debts()[0]
        debtor.set_grade(subject, 6)
        print(f"\nПосле пересдачи «{subject}» у {debtor.get_full_name()}:")
        print(debtor.report())

    avg = sum(st.average() for st in students) / len(students)
    print(f"\nСредний балл по всем студентам: {avg:.2f}")


# ===========================================================================
# Задание 4. Свой класс: методы экземпляра, статические и методы класса (1–3 балла)
# ===========================================================================
class Temperature:
    """Температура. Внутри всегда хранится в градусах Цельсия."""

    count = 0                 # атрибут класса: сколько объектов создано всего
    ABSOLUTE_ZERO = -273.15

    def __init__(self, celsius):
        if celsius < Temperature.ABSOLUTE_ZERO:
            raise ValueError("ниже абсолютного нуля не бывает")
        self.celsius = celsius
        Temperature.count += 1

    # ---- методы экземпляра: первый параметр self, работают с данными объекта ----
    def to_fahrenheit(self):
        return Temperature.c_to_f(self.celsius)

    def to_kelvin(self):
        return self.celsius - Temperature.ABSOLUTE_ZERO

    def describe(self):
        if self.celsius <= 0:
            return "мороз"
        if self.celsius < 15:
            return "прохладно"
        if self.celsius < 25:
            return "тепло"
        return "жарко"

    def __str__(self):
        return f"{self.celsius:.1f} °C"

    # ---- статический метод: не получает ни self, ни cls.
    # Обычная функция, положенная внутрь класса, потому что по смыслу к нему относится.
    @staticmethod
    def c_to_f(c):
        return c * 9 / 5 + 32

    @staticmethod
    def f_to_c(f):
        return (f - 32) * 5 / 9

    # ---- методы класса: первый параметр cls, то есть сам класс.
    # Часто служат «альтернативными конструкторами».
    @classmethod
    def from_fahrenheit(cls, f):
        return cls(cls.f_to_c(f))

    @classmethod
    def from_kelvin(cls, k):
        return cls(k + cls.ABSOLUTE_ZERO)

    @classmethod
    def created(cls):
        return cls.count


def task4():
    t1 = Temperature(22)                     # обычный конструктор
    t2 = Temperature.from_fahrenheit(14)     # метод класса
    t3 = Temperature.from_kelvin(310.15)     # метод класса

    for t in (t1, t2, t3):
        print(f"{t}: {t.to_fahrenheit():.1f} °F, {t.to_kelvin():.2f} K, {t.describe()}")

    print("\nСтатический метод без создания объекта: 100 °C =",
          Temperature.c_to_f(100), "°F")
    print("Метод класса: создано объектов Temperature:", Temperature.created())

    try:
        Temperature(-300)
    except ValueError as e:
        print("Проверка ошибки: Temperature(-300) ->", e)


# ===========================================================================
TASKS = {
    "1": ("Класс Square", task1),
    "2": ("Stationery и наследники", task2),
    "3": ("Faculty и Student", task3),
    "4": ("Свой класс Temperature", task4),
}


def main():
    while True:
        print("\n===== ЛР1, часть 3, вариант 4 =====")
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
