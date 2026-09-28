import { configureStore } from '@reduxjs/toolkit'
import { useDispatch, useSelector } from 'react-redux'
import searchReducer from './searchSlice'
import uiReducer from './uiSlice'

export const store = configureStore({
  reducer: {
    search: searchReducer,
    ui: uiReducer,
  },
  middleware: (getDefault) =>
    // Phase results can be large; skip the dev-only deep checks on them.
    getDefault({ immutableCheck: false, serializableCheck: false }),
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export const useAppDispatch = useDispatch.withTypes<AppDispatch>()
export const useAppSelector = useSelector.withTypes<RootState>()
