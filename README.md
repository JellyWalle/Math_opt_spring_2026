# Math_opt_spring_2026


## Требования

- **Python** ≥ 3.8
- **Conda** (рекомендуется) или `pip`
- Зависимости:
  - `numpy`
  - `matplotlib`
  - `setuptools`

## 1. Установка

### 1.1 Клонирование репозитория

```bash
git clone https://github.com/your-username/Math_opt_spring_2026.git
cd Math_opt_spring_2026

# Инициализация субмодуля с библиотекой CEC2017
git submodule update --init --recursive
```

### 1.2 Создание окружения conda:
```bash
conda create -n opt_env python=3.10 numpy matplotlib setuptools -y
conda activate opt_env
```

### 1.3 Устновка пакета CEC2017:
```bash
pip install -e ./cec2017-py
```

## 2.Визуализация функций из бенчмарка **CEC 2017** 
```bash
python src/plot_f1_f10.py
```
Результаты сохраняются в папку **plots**