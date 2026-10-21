"""Копия apps/vehicles/plates.py из sewing-factory: камера и сервер должны приводить номера к одному виду.
Меняешь здесь - поменяй и там (и наоборот).
"""
import re

# Русские буквы, похожие на латинские (на номерах РФ только они)
CYRILLIC_TO_LATIN = str.maketrans('АВЕКМНОРСТУХавекмнорстух', 'ABEKMHOPCTYXABEKMHOPCTYX')
NOT_ALNUM = re.compile(r'[^A-Z0-9]')

# Типичные ошибки распознавания: буква на месте цифры и наоборот
TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
RU_LETTERS = set('ABEKMHOPCTYX')

FORMATS = (
    ('KG', r'^(0[1-9]|1[01])\d{3}[A-Z]{3}$'),     # 01 123 ABC
    ('KG_ORG', r'^(0[1-9]|1[01])\d{3}[A-Z]{2}$'),  # 01 857 BF
    ('KG_OLD', r'^[ABCDEFHINOSTZ]\d{4}[A-Z]{1,2}$'),  # B 7777 BB
    ('KZ', r'^\d{3}[A-Z]{2,3}(0[1-9]|1\d|20)$'),  # 123 ABC 02
    ('KZ_OLD', r'^[A-Z]\d{3}[A-Z]{2,3}$'),  # A 123 BCD
    ('RU', r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}(\d{2}|[1-9]\d{2})$'),  # A 123 BC 77
)

# Самые частые форматы можно исправлять по позициям символов.
# D - цифра, L - любая латинская буква, R - одна из 12 букв номеров РФ
SLOTS = (('KG', 'DDDDDLLL'), ('KG_ORG', 'DDDDDLL'), ('KZ', 'DDDLLLDD'), ('KZ', 'DDDLLDD'),
         ('RU', 'RDDDRRDD'), ('RU', 'RDDDRRDDD'), ('KG_OLD', 'LDDDDLL'), ('KG_OLD', 'LDDDDL'),
         ('KZ_OLD', 'LDDDLLL'), ('KZ_OLD', 'LDDDLL'))
MAX_FIXES = 2  # больше двух замен - уже не опечатка, а другой номер (например, именной)

# Пометка страны, которую камера или человек могли включить в номер, и какие форматы с ней бывают.
# KG после кода региона бывает только на новых номерах, а слева от номера - и на старых
COUNTRY_TAGS = (
    (r'^(..)KG(.{5,6})$', {'KG', 'KG_ORG'}),
    (r'^KG(.{6,8})$', {'KG', 'KG_ORG', 'KG_OLD'}),
    (r'^(.{7,8})KG$', {'KG', 'KG_ORG'}),
    (r'^KZ(.{6,8})$', {'KZ', 'KZ_OLD'}),
    (r'^(.{8,9})RUS$', {'RU'}),
)


def plate_format(plate):
    for name, pattern in FORMATS:
        if re.match(pattern, plate):
            return name
    return ''


def _fix_by_slots(plate, slots):
    fixed, fixes = [], 0
    for char, slot in zip(plate, slots):
        if slot == 'D' and not char.isdigit():
            char, fixes = TO_DIGIT.get(char), fixes + 1
        elif slot in 'LR' and char.isdigit():
            char, fixes = TO_LETTER.get(char), fixes + 1
        if char is None or (slot == 'R' and char not in RU_LETTERS):
            return None, fixes
        fixed.append(char)
    return ''.join(fixed), fixes


def simplify_plate(text):
    """Только заглавные латинские буквы и цифры: '01 kg 123 авс' -> '01KG123ABC'."""
    return NOT_ALNUM.sub('', ('' if text is None else str(text)).upper().translate(CYRILLIC_TO_LATIN))


def plate_candidates(text):
    """Все правдоподобные прочтения номера, самое вероятное первым: '01 KG 123 АВС' -> ['01123ABC'].

    Прочтений бывает несколько, когда номер подходит под разные форматы: 'O1857BF' - это и старый ошский
    номер, и '01 857 BF' с буквой O вместо нуля.
    """
    plate = simplify_plate(text)
    variants = [(plate, None, -2)]
    for pattern, allowed in COUNTRY_TAGS:
        match = re.match(pattern, plate)
        if match:
            variants.append((''.join(match.groups()), allowed, -1))
    found = []
    for variant, allowed, rank in variants:
        name = plate_format(variant)
        if name and (allowed is None or name in allowed):
            found.append((0, rank, variant))
        for order, (slot_name, slots) in enumerate(SLOTS):  # при равенстве побеждает формат выше в списке
            if len(variant) == len(slots) and (allowed is None or slot_name in allowed):
                fixed, fixes = _fix_by_slots(variant, slots)
                if fixed and 0 < fixes <= MAX_FIXES and plate_format(fixed) == slot_name:
                    found.append((fixes, order, fixed))
    candidates = []
    for _, _, candidate in sorted(found):
        if candidate not in candidates:
            candidates.append(candidate)
    # Старый номер 'O 1857 BF' (O, I, D - коды областей) не отличить от нового '01 857 BF', прочитанного
    # с ошибкой. Новых номеров больше, поэтому такой номер всегда пишем как новый: тогда любое прочтение
    # одной машины - с пометкой KG или без, с буквой O или с нулём - даёт один и тот же номер
    new_format = [c for c in candidates if plate_format(c) in ('KG', 'KG_ORG')]
    if new_format and plate_format(candidates[0]) == 'KG_OLD':
        candidates.remove(new_format[0])
        candidates.insert(0, new_format[0])
    return candidates or [plate]


def normalize_plate(text):
    """'01 KG 123 АВС' -> '01123ABC'. Если номер не похож ни на один формат, возвращает его как есть."""
    return plate_candidates(text)[0]


class PlateError(ValueError):
    pass


SEPARATORS = re.compile(r'[\s\-_.|]')


def clean_plate(text):
    """Как normalize_plate, но не пропускает мусор: чужие буквы и слишком короткие или длинные номера."""
    raw = SEPARATORS.sub('', ('' if text is None else str(text)).upper().translate(CYRILLIC_TO_LATIN))
    if NOT_ALNUM.search(raw):
        raise PlateError('В номере могут быть только цифры и латинские буквы (или похожие на них русские)')
    plate = normalize_plate(raw)
    if not 2 <= len(plate) <= 12:
        raise PlateError('В номере должно быть от 2 до 12 символов')
    return plate
