CHOICES = {
    "🇺🇸 🚺 Maple ❤️": "af_maple",
    "🇺🇸 🚺 Sol 🔥": "af_sol",
    "🇬🇧 🚺 Vale 🎧": "bf_vale",
}


# 定义不连续的数字范围
zf_ranges = [
    range(1, 9),
    range(17, 20),
    range(21, 25),
    range(26, 29),
    range(32, 33),
    range(36, 37),
    range(38, 41),
    range(42, 45),
    range(46, 50),
    range(51, 52),
    range(59, 60),
    range(60, 61),
    range(67, 68),
    range(70, 80),
    range(83, 89),
    range(90, 91),
    range(92, 95),
    range(99, 100),
]
zm_ranges = [
    range(9, 17),
    range(20, 21),
    range(25, 26),
    range(29, 32),
    range(33, 36),
    range(37, 38),
    range(41, 42),
    range(45, 46),
    range(50, 51),
    range(52, 59),
    range(61, 67),
    range(68, 70),
    range(80, 83),
    range(89, 90),
    range(91, 92),
    range(95, 99),
    range(100, 101),
]



# 使用列表推导式为中国的选项添加到字典中
CHOICES.update({f"🇨🇳 🚹 {i:03d}": f"zf_{i:03d}" for r in zf_ranges for i in r})
CHOICES.update({f"🇨🇳 🚺 {i:03d}": f"zm_{i:03d}" for r in zm_ranges for i in r})


# 使用循环语句为中国的选项添加到字典中
""" for r in zf_ranges:
    for i in r:
        key = f"🇨🇳 🚹 {i:03d}"  # 格式化字符串，确保数字是三位数
        value = f"zf_{i:03d}"  # 格式化字符串，确保数字是三位数
        CHOICES[key] = value
for r in zm_ranges:
    for i in r:
        key = f"🇨🇳 🚺 {i:03d}"  # 格式化字符串，确保数字是三位数
        value = f"zm_{i:03d}"  # 格式化字符串，确保数字是三位数
        CHOICES[key] = value """

# 提取英文选项的值
en_choices = [value for key, value in CHOICES.items() if key.startswith(('🇺🇸 🚹 ', '🇺🇸 🚺 ', '🇬🇧 🚹 ', '🇬🇧 🚺 '))]

# 提取中文选项的值
zh_choices = [value for key, value in CHOICES.items() if key.startswith(('🇨🇳 🚹 ', '🇨🇳 🚺 '))]

# 按照指定格式打印结果
print(f'en: {en_choices}')
print(f'zh: {zh_choices}')