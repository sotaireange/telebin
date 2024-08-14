from aiogram import Router
from aiogram.filters import Command,StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message
from aiogram.fsm.state import State, StatesGroup
import asyncio
from keyboard import main_keyboard,run_keyboard
from binance_bot import AlgoBot

class Main(StatesGroup):
    RUN = State()
    UNRUN = State()
    SET = State()


router=Router()



async def get_text(state:FSMContext):
    data=await state.get_data()
    run=data.get('run',0)
    api=data.get('api',"---")
    secret=data.get('secret',"---")
    atr=data.get('atr',"---")
    key_value=data.get('key_value',"---")
    coin=data.get('coin',"---")
    bal=data.get('bal',"---")
    leverage=data.get('leverage',"---")
    time_frame=data.get('time_frame',"---")
    tp=data.get('tp','---')
    sl=data.get('sl','---')
    ha=data.get('hiken_ashi',0)
    if ha:
        hiken_ashi="Включен"
    else:
        hiken_ashi="Выключен"

    text=(f'API - <{api}>\n'
          f'Secret - <{secret}>\n'
          f'COIN - <{coin}>\n'
          f'Time Frame - <{time_frame}>\n'
          f'ATR Period- <{atr}>\n'
          f'Key Value - <{key_value}>\n'
          f'Balance - <{bal}>\n'
          f'Leverage - <{leverage}>\n'
          f'Тейк Профит - <{tp}>\n'
          f'Стоп Лосс - <{sl}>\n'
          f'Heikin Ashi - {hiken_ashi}\n')

    return text


@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    text=await get_text(state)
    data=await state.get_data()
    run=data.get('run',0)
    if run==0:
        await message.answer(text=text,reply_markup=main_keyboard())
        await message.delete()
    else:
        text="Бот работает\n" + text
        await message.answer(text=text,reply_markup=run_keyboard())
        await message.delete()


@router.message(StateFilter(Main.SET))
async def set_data(message: Message, state: FSMContext):
    data= await state.get_data()
    key=data.get('key')
    text=message.text
    await state.update_data({key:text})
    text_full=await get_text(state)
    text=f'Данные успешно сохранены <{key}> - <{text}>\n' + text_full
    await message.answer(text=text,reply_markup=main_keyboard())
    await state.set_state(Main.UNRUN)

@router.callback_query(lambda call: call.data=="hiken_ashi")
async def hiken_ashi(call: CallbackQuery, state: FSMContext):
    data= await state.get_data()
    ha=data.get('hiken_ashi',0)
    if ha:
        ha=0
    else:
        ha=1
    await state.update_data({"hiken_ashi":ha})
    text= await get_text(state)
    await call.message.edit_text(text=text,reply_markup=main_keyboard())

@router.callback_query(lambda call: call.data=="run")
async def run(call: CallbackQuery, state: FSMContext):
    data= await state.get_data()
    bot=AlgoBot(data)
    await call.message.edit_text(text='Бот работает',reply_markup=run_keyboard())
    await asyncio.sleep(10)
    await state.update_data({'run':1})
    await state.set_state(Main.RUN)
    await bot.start_trade(state,call.message)


@router.callback_query(lambda call: call.data=="unrun")
async def unrun(call: CallbackQuery, state: FSMContext):
    await state.set_state(Main.UNRUN)
    await state.update_data({'run': 0})
    await start(call.message,state)

@router.callback_query(lambda call: True)
async def set_state_for_data(call: CallbackQuery, state: FSMContext):
    await state.update_data({'key':call.data})
    await call.message.edit_text(text=f'Введите новое значение для <{call.data}>')
    await state.set_state(Main.SET)

