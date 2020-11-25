"""

This module validates the various YAML files within the project. It uses the yamale library.
"""
from collections import Mapping
from packaging import version
from urllib.error import HTTPError

import yamale
from yamale.validators import DefaultValidators, Validator
from yamale.validators import constraints as con

import dwd_dl.cfg as cfg


class URLValidator(Validator):
    """ Custom url validator """
    tag = 'url'

    def _is_valid(self, value):
        try:
            return cfg.check_connection(url=value)
        except HTTPError:
            return False

    def fail(self, value):
        return f"{value} is not a valid URL."


class RangesDateFormatValidator(Validator):
    """ Custom Ranges Date Format validator"""
    tag = 'ranges_date_format'

    def _is_valid(self, value):
        stamps_ranges = ('%Y', '%m', '%d', '%H', '%M', '%S')
        return all(stamp in value for stamp in stamps_ranges)


class TimestampDateFormatValidator(Validator):
    """ Custom Timestamp Date Format validator """
    tag = 'timestamp_date_format'

    def _is_valid(self, value):
        stamps_timestamps = ('%y', '%m', '%d', '%H', '%M')
        return all(stamp in value for stamp in stamps_timestamps)


class CustomDateMapValidator(Validator):
    """ A custom map validator for handling start and end date"""
    tag = 'custom_date_map'
    constraints = [con.Key, con.LengthMax, con.LengthMin]

    def __init__(self, *args, **kwargs):
        super(CustomDateMapValidator, self).__init__(*args, **kwargs)
        self.validators = [val for val in args if isinstance(val, Validator)]

    def validate(self, value):
        """
        Check if ``value`` is valid.
        :returns: [errors] If ``value`` is invalid, otherwise [].
        """
        errors = []

        # Make sure the type validates first.
        valid = self._is_valid(value)
        if not valid:
            errors.append(self.fail(value))
            return errors

        # Then validate all the constraints second.
        for constraint in self._constraints_inst:
            error = constraint.is_valid(value)
            if error:
                if isinstance(error, list):
                    errors.extend(error)
                else:
                    errors.append(error)

        failed_validation = False
        for val in self.validators:
            for key in value:
                error = val.validate(value[key])
                if bool(error):
                    failed_validation = True
                errors.extend(error)

        # Additional only to this custom class: compare if the two dates are in the correct order
        if not failed_validation:
            try:
                if value['MAX_END_DATE'] < value['MIN_START_DATE']:
                    message = f"MIN_START_DATE {value['MIN_START_DATE']} is greater " \
                              f"than MAX_END_DATE {value['MAX_END_DATE']}"
                    errors.append(message)
            except KeyError:
                pass

        return errors

    def _is_valid(self, value):
        return isinstance(value, Mapping)  # TODO: what the hell? It doesn't look i is in max min timestamp.


class Version(Validator):
    """ Custom version validator """
    tag = 'version'

    def _is_valid(self, value):
        try:
            version.Version(value)
            return True
        except version.InvalidVersion:
            return False


def validate(path=None):
    validators = DefaultValidators.copy()
    for validator in [
        URLValidator, RangesDateFormatValidator, TimestampDateFormatValidator, CustomDateMapValidator, Version
    ]:
        validators[validator.tag] = validator
    schema = yamale.make_schema(cfg.path_to_resources_folder(filename='RADOLAN_CFG_SCHEMA.yml'), validators=validators)
    data = yamale.make_data(path=path)
    yamale.validate(schema, data)
