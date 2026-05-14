# Math_opt_spring_2026


## Требования

- **Python** ≥ 3.8
- **Conda** (рекомендуется) или `pip`
- Зависимости:
  - `numpy`
  - `matplotlib`
  - `setuptools`
  - `pandas`
  - `scipy`
  - `pyswarms`
  - `pygad`
  - `scikit_posthocs`

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
conda create -n opt_env python=3.10 numpy matplotlib setuptools pandas scipy pyswarms pygad -y
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

## 3.1 Оптимизация 10-мерных случаев (scipy)
```bash
python src/optimize_scipy.py
```
Результаты сохраняются в папку **results/optimization_l_bfgs_b.csv**

## 3.2 Оптимизация 10-мерных случаев (PSO)
```bash
python src/optimize_pso.py
```
Результаты сохраняются в папку **results/optimization_pso.csv**

## 3.3 Оптимизация 10-мерных случаев (GA)
```bash
python src/optimize_ga.py
```
Результаты сохраняются в папку **results/optimization_ga.csv**


