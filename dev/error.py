# Error module for defining error messages and how they are handled

class ErrorCheck:
    def __init__(self):
        self.allowed_parameter_values = {
            'zscore':{'type':bool,'values':[True,False]},
            'whiten':{'type':bool,'values':[True,False]},
            'verbose':{'type':bool,'values':[True,False]},
            'float_example':{'type':float,'values':[0, float('inf')]}
        }
    def raise_error(self, error_type, parameter=None, parameter_value=None):
        if error_type == 'parameter value':
            param_info = self.allowed_parameter_values[parameter]
            raise Exception(
                    'Wrong parameter value (%s) provided for parameter %s'%(parameter_value, parameter),
                    'Parameter should be of type %s and in the interval OR in the set [%s, %s]'%(
                        param_info['type'], 
                        param_info['values'][0], 
                        param_info['values'][1])   
            )

    def is_param_value_correct(self, parameter, parameter_value):
        param_info = self.allowed_parameter_values[parameter]
        if param_info['type'] == bool:
            return parameter_value in param_info['values']
        elif param_info['type'] == double:
            return parameter_value >= param_info['values'][0] and parameter_value <= param_info['values'][1]
        elif param_info['type'] == int:
            return type(param_info['type']) == int and parameter_value >= param_info['values'][0] and parameter_value <= param_info['values'][1]
        else:
            return False

    def check_all_parameters(self, parameter_dict):
        for k,v in parameter_dict.items():
            if not (self.is_param_value_correct(k, v)):
                self.raise_error('parameter value', k, v)