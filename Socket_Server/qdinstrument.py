import win32com.client
import pythoncom


class QDInstrument:
    def __init__(self, instrument_type):
        instrument_type = instrument_type.upper()
        if instrument_type == 'DYNACOOL':
            self._class_id = 'QD.MULTIVU.DYNACOOL.1'
        elif instrument_type == 'PPMS':
            self._class_id = 'QD.MULTIVU.PPMS.1'
        elif instrument_type == 'VERSALAB':
            self._class_id = 'QD.MULTIVU.VERSALAB.1'
        elif instrument_type == 'MPMS3':
            self._class_id = 'QD.MULTIVU.MPMS3.1'
        else:
            raise Exception('Unrecognized instrument type: {0}.'.format(instrument_type))
        self._mvu = win32com.client.Dispatch(self._class_id)

    def set_temperature(self, temperature, rate, mode):
        """Sets temperature and returns MultiVu error code"""
        err = self._mvu.SetTemperature(temperature, rate, mode)
        return err

    def get_temperature(self):
        """Gets and returns temperature info as (MultiVu error, temperature, status)"""
        arg0 = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_R8, 0.0)
        arg1 = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        err = self._mvu.GetTemperature(arg0, arg1)
        # win32com reverses the arguments, so:
        return err, arg1.value, arg0.value

    def set_field(self, field, rate, approach, mode):
        """Sets field and returns MultiVu error code"""
        err = self._mvu.SetField(field, rate, approach, mode)
        return err

    def get_field(self):
        """Gets and returns field info as (MultiVu error, field, status)"""
        arg0 = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_R8, 0.0)
        arg1 = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        err = self._mvu.GetField(arg0, arg1)
        # win32com reverses the arguments, so:
        return err, arg1.value, arg0.value

    def set_chamber(self, code):
        """Sets chamber and returns MultiVu error code"""
        err = self._mvu.SetChamber(code)
        return err

    def get_chamber(self):
        """Gets chamber status and returns (MultiVu error, status)"""
        arg0 = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        err = self._mvu.GetChamber(arg0)
        return err, arg0.value
