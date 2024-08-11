from aiogram import types
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def main_keyboard():
    markup=InlineKeyboardBuilder()
    markup.row(types.InlineKeyboardButton(text='Включить', callback_data='run'))
    markup.row(types.InlineKeyboardButton(text='API', callback_data='api'))
    markup.row(types.InlineKeyboardButton(text='Secret', callback_data='secret'))
    markup.row(types.InlineKeyboardButton(text='Монетка', callback_data='coin'))
    markup.row(types.InlineKeyboardButton(text='Time Frame', callback_data='time_frame'))
    markup.row(types.InlineKeyboardButton(text='Key Value', callback_data='key_value'))
    markup.row(types.InlineKeyboardButton(text='ATR Period', callback_data='atr'))
    markup.row(types.InlineKeyboardButton(text='Balance', callback_data='bal'))
    markup.row(types.InlineKeyboardButton(text='Leverage', callback_data='leverage'))
    markup.row(types.InlineKeyboardButton(text='Тейк-Профит',callback_data='tp'))
    markup.row(types.InlineKeyboardButton(text='Стоп-Лосс',callback_data='sl'))

    return markup.as_markup()


def run_keyboard():
    markup=InlineKeyboardBuilder()
    markup.row(types.InlineKeyboardButton(text='Выключить',callback_data='unrun'))
    return markup.as_markup()
