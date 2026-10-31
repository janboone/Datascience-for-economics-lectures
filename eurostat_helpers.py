"""
Helper utilities for downloading and reshaping Eurostat data.

Typical usage
-------------
    from eurostat_helpers import get_eurostat, EU_countries

    # GDP per capita, current prices, EU countries, 2011-2020
    df_gdp = get_eurostat(
        code='nama_10_pc',
        value_name='GDP per capita',
        filter_dict={'unit': 'CP_EUR_HAB', 'na_item': 'B1GQ'},
        years=range(2011, 2021),
    )

    # BMI variable (by sex, percentages → fractions)
    df_bmi = get_eurostat(
        code='hlth_ehis_bm1i',
        id_vars=['sex'],
        value_name='high bmi',
        filter_dict={
            'bmi': 'BMI_GE30',
            'quant_inc': 'TOTAL',
            'age': 'Y15-64',
            'sex': ['F', 'M'],
        },
        years=[2014, 2019],
        divide_by=100,
    )
"""

import pandas as pd
from country_codes import eurostat_dictionary

EU_countries = [
    'Belgium', 'Bulgaria', 'Czechia', 'Denmark',
    'Germany', 'Estonia', 'Ireland', 'Greece', 'Spain', 'France',
    'Croatia', 'Italy', 'Cyprus', 'Latvia', 'Lithuania', 'Luxembourg',
    'Hungary', 'Malta', 'Netherlands', 'Austria', 'Poland', 'Portugal',
    'Romania', 'Slovenia', 'Slovakia', 'Finland', 'Sweden',
    'United Kingdom', 'Iceland', 'Liechtenstein', 'Norway', 'Switzerland',
    'Bosnia and Herzegovina',
]


def get_eurostat(
    code,
    id_vars=None,
    value_name='value',
    filter_dict=None,
    years=None,
    countries=None,
    divide_by=1.0,
):
    """
    Fetch a Eurostat dataset and reshape it to long format.

    This function handles the steps that appear in every Eurostat download:
    renaming the geo column, translating country codes to names, filtering
    rows, melting year columns into a single 'year' column, and selecting
    countries.

    Parameters
    ----------
    code : str
        Eurostat dataset code, e.g. ``'nama_10_pc'``.  Find it in the
        Eurostat data browser — the code appears as
        "online data code: <code>" at the top-left of any dataset page.

    id_vars : list of str, optional
        Extra categorical columns to keep alongside ``'country'``, for
        example ``['sex']`` when the variable is broken down by gender.
        ``'country'`` is always included automatically.  Defaults to no
        extra columns (country + year only).

    value_name : str, optional
        Name for the value column in the output, e.g. ``'GDP per capita'``
        or ``'high bmi'``.  Defaults to ``'value'``.

    filter_dict : dict, optional
        Row filters to apply before reshaping.  Each key is a column name
        and the value is either a single accepted value or a list of
        accepted values.  Rows not matching are dropped.

        Example::

            filter_dict={
                'unit': 'CP_EUR_HAB',   # single value
                'sex':  ['F', 'M'],     # list of values
            }

    years : iterable of int, optional
        Years to include, e.g. ``range(2011, 2021)`` or ``[2014, 2019]``.
        If ``None``, all four-digit year columns found in the raw data are
        used.

    countries : list of str, optional
        Country names to keep in the output.  Defaults to ``EU_countries``
        (the 33 EU/EEA countries used in the lecture).

    divide_by : float, optional
        Divide all values by this number.  Use ``100`` to convert a
        percentage variable (0–100) to a fraction (0–1).  Defaults to
        ``1.0`` (no change).

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with columns::

            ['country'] + id_vars + ['year', value_name]

        ``'year'`` is an integer column.

    Examples
    --------
    GDP per capita (one row per country-year):

    >>> df = get_eurostat(
    ...     code='nama_10_pc',
    ...     value_name='GDP per capita',
    ...     filter_dict={'unit': 'CP_EUR_HAB', 'na_item': 'B1GQ'},
    ...     years=range(2011, 2021),
    ... )
    >>> df.head()
       country  year  GDP per capita
    0  Austria  2011         36970.0
    1  Belgium  2011         34060.0
    ...

    BMI variable broken down by sex and two survey years:

    >>> df = get_eurostat(
    ...     code='hlth_ehis_bm1i',
    ...     id_vars=['sex'],
    ...     value_name='high bmi',
    ...     filter_dict={
    ...         'bmi': 'BMI_GE30',
    ...         'quant_inc': 'TOTAL',
    ...         'age': 'Y15-64',
    ...         'sex': ['F', 'M'],
    ...     },
    ...     years=[2014, 2019],
    ...     divide_by=100,
    ... )
    """
    import eurostat as _eurostat

    if id_vars is None:
        id_vars = []
    if countries is None:
        countries = EU_countries

    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    df = _eurostat.get_data_df(code)

    # ── 2. Rename the geo\TIME_PERIOD column ──────────────────────────────────
    # The column name uses a backslash and changes slightly across datasets;
    # we find it by its 'geo' prefix.
    geo_col = next(c for c in df.columns if c.startswith('geo'))
    df = df.rename({geo_col: 'geo'}, axis=1)

    # ── 3. Translate country codes to full names ───────────────────────────────
    df['country'] = df['geo'].replace(eurostat_dictionary)

    # ── 4. Filter rows ────────────────────────────────────────────────────────
    if filter_dict:
        for col, val in filter_dict.items():
            if isinstance(val, (list, tuple)):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    # ── 5. Identify year columns ──────────────────────────────────────────────
    def _is_year(c):
        try:
            return 1990 <= int(c) <= 2100
        except (ValueError, TypeError):
            return False

    available_years = [c for c in df.columns if _is_year(c)]
    if years is not None:
        year_cols = [str(y) for y in years if str(y) in available_years]
    else:
        year_cols = available_years

    # ── 6. Keep only the columns we need, then melt ───────────────────────────
    keep = ['country'] + id_vars + year_cols
    df = df[[c for c in keep if c in df.columns]]

    df = pd.melt(
        df,
        id_vars=['country'] + id_vars,
        value_vars=year_cols,
        var_name='year',
        value_name=value_name,
    )

    # ── 7. Clean up ───────────────────────────────────────────────────────────
    df['year'] = df['year'].astype(int)
    df = df[df['country'].isin(countries)].reset_index(drop=True)

    if divide_by != 1.0:
        df[value_name] = df[value_name].astype(float) / divide_by

    return df
