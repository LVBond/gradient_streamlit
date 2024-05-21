import numpy as np 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.title('Градиентный спуск')

# Пример вывода текста и данных
st.write('Приложение Streamlit для визуализации градиентного спуска с возможностью выбора learning rate')


#loss  - возвращает значение функции в точке
def loss(w):
    return w ** 2 + 3

#derivative - возвращает значение производной функции в точке
def derivative(x):
    deriv = 2 * x
    return deriv

#step - новое значение координаты и значение функции в этой точке
def step(x, alpha):
    ''' alpha - learning rate
        coord - x_coord
        loss_value - y_coord '''
    coord = x - alpha * derivative(x)
    loss_value = loss(coord)
    return coord, loss_value

#get_minima - возвращает список иксов и список их значений функции. eps - критерий останова, alpha - скорость спуска (learning rate)
def get_minima(x_0, eps, alpha): 
    x_current = x_0
    loss_current = loss(x_0)
    
    x = []
    loss_values = []
    
    x.append(x_current)
    loss_values.append(loss_current)
    
    while True:
        x_new, loss_new = step(x_current, alpha)
        if abs(loss_new - loss_current) < eps:
            print('x координата - ', x_current)
            print('Наименьшее значение функции - ', loss_current)
            break
        else:
            x.append(x_new)
            loss_values.append(loss_new)
            x_current, loss_current = x_new, loss_new
    return np.array(x), np.array(loss_values)


#alpha = 0.1
alpha = st.slider('Выберите значение learning rate (alpha)', 0.01, 0.5, 0.1, 0.01)
eps = 0.0001

x_vals, loss_vals = get_minima(2, eps, alpha)
print(x_vals[-1])
print(loss_vals[-1])

# График функции `loss` - движение градиентного спуска. По координатам, возвращенным функцией `get_minima`
x = np.linspace(-4, 4, 1000)
x_vals, loss_vals = get_minima(2, eps, alpha)


# Создание графика
fig, ax = plt.subplots()
ax.plot(x, loss(x), color='blue')
ax.plot(x_vals, loss_vals, color='red', marker='o')
ax.set_title(f"learning_rate = {alpha}")
ax.grid()

# Отображение графика в Streamlit
st.pyplot(fig)




# Создаем ползунок для настройки значения alpha
#alpha_value = st.slider("Настройка скорости спуска (learning rate)", 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)

# Отображаем выбранное значение alpha
st.write(f"Выбранное значение alpha: ")


