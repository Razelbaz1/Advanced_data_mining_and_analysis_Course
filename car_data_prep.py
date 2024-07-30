import numpy as np
import pandas as pd
import re

def remove_Duplicates(data):
    data.drop_duplicates(inplace=True)
    return data

def standardize_manufactor(name):
    # ... (standardize_manufactor function content)
    return name

def standardize_column_regex(df, column):
    df[column] = df[column].apply(standardize_manufactor)
    return df

def remove_manufactor_name_in_model_col(data):
    for index, row in data.iterrows():
        manufactor = str(row['manufactor'])
        model = str(row['model'])
        if manufactor in model:
            data.at[index, 'model'] = model.replace(manufactor, '').strip()
    return data

def clean_model_name(model_name):
    model_name = re.sub(r'[\r\n\t]', ' ', model_name)
    model_name = re.sub(r'\(.*?\)', '', model_name)
    model_name = re.sub(r'\s+', ' ', model_name).strip()
    model_name = re.sub(r'ה?חדש(ה)?', '', model_name).strip()
    model_name = re.sub(r'ה?סדרה?', '', model_name).strip()
    # Removed the erroneous reference to data['model']
    model_name = re.sub(r'\bCivic\b', 'סיוויק', model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bSX4\b', 'SX4', model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bS6\b', 'S6', model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bOne\b', 'ONE', model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bJuke\b', "ג'וק'", model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bJazz\b', "ג'אז", model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bGolf\b', 'גולף', model_name, flags=re.IGNORECASE)
    model_name = re.sub(r'\bFocus\b', 'פוקוס', model_name, flags=re.IGNORECASE)
    model_name = re.sub(r"\bג'וק ג'וק\b", "ג'וק", model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(SX4 קרוסאובר|קרוסאובר)\b', 'SX4', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bנירו ?(ev|phev)?\b', 'נירו', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bמוקה x\b', 'מוקה', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(סיוויק הייבריד|סיוויק סדאן|סיוויק סטיישן|סיוויק האצ\'בק|סיוויק|סיוויק האצ’בק)\b', 'סיוויק', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(ג\'אז הייבריד|ג\'אז|ג\'אז הייבריד|ג\'אז|ג\'אז)\b', 'ג\'אז', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(accord|אקורד)\b', 'ACCORD', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(2|3|5|6)\b', r'\1', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(108|208|308|508|5008|2008)\b', r'\1', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(fluence|קליאו|מגאן|25|גרנד סניק|קפצ\'ור|פלואנס)\b', r'\1', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(קאנטרימן|קאונטרימן)\b', 'קאנטרימן', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(מיטו / mito|מיטו|ג\'ולייטה)\b', 'מיטו', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(וויאג`ר|גראנד,? וויאג\'ר)\b', 'גראנד וויאג\'ר', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\b(g|ג)\s*וויאג\'ר\b', 'גראנד וויאג\'ר', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bE[-\s]?class\b', 'E-Class', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bC[-\s]?class\b', 'C-Class', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bC-Class Taxi\b', 'C-Class', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bE-Class קופה / קבריולט\b', 'E-Class', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\bC-Class קופה\b', 'C-Class', model_name, flags= re.IGNORECASE)
    model_name = re.sub(r'\be- class\b', 'E-Class', model_name, flags= re.IGNORECASE)
    return model_name

def normalize_model_name(model_name):
    return re.sub(r'\s+', ' ', model_name.lower()).strip()

def bin_year_column(data, num_bins=5):
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = data['Year'].quantile(quantiles).unique()
    bin_edges = np.unique(bin_edges)
    bin_edges[-1] = data['Year'].max() + 1
    labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges) - 1)]
    data['Year_range'] = pd.cut(data['Year'], bins=bin_edges, labels=labels, include_lowest=True, right=False)
    return data

def handle_hand_column(data):
    median_hand = data['Hand'].median()
    if not median_hand.is_integer():
        median_hand = np.ceil(median_hand)
    data['Hand'] = data['Hand'].fillna(median_hand)
    return data

def standardize_gear(gear):
    gear_mapping = {
        'automatic': 'אוטומטי',
        'auto': 'אוטומטי',
        'אוטומטית': 'אוטומטי',
        'אוטומט': 'אוטומטי',
        'tiptronic': 'טיפטרוניק',
        'manual': 'ידני',
        'ידני': 'ידני',
        'robotic': 'רובוטי',
        'רובוטית': 'רובוטי',
        'undefined': np.nan,
        'לא מוגדר': np.nan
    }
    if pd.isna(gear):
        return gear
    gear = gear.lower().strip()
    return gear_mapping.get(gear, gear)

def handle_gear_column(data):
    data['Gear'] = data['Gear'].apply(standardize_gear)
    most_common_gear = data['Gear'].mode()[0]
    data['Gear'] = data['Gear'].fillna(most_common_gear)
    data['Auto_Gear'] = (data['Gear'] != 'אוטומטי').astype(int).astype('category')
    return data

def standardize_engine_type(engine_type):
    engine_type_mapping = {
        'gasoline': 'בנזין',
        'diesel': 'דיזל',
        'turbo diesel': 'טורבו דיזל',
        'gas': 'גז',
        'hybrid': 'היברידי',
        'היבריד': 'היברידי',
        'היברידי': 'היברידי',
        'electrical': 'חשמלי',
        'חשמלי': 'חשמלי',
        'undefined': np.nan,
        'לא מוגדר': np.nan
    }
    if pd.isna(engine_type):
        return engine_type
    engine_type = engine_type.lower().strip()
    return engine_type_mapping.get(engine_type, engine_type)

def handle_engine_type_column(data):
    data['Engine_type'] = data['Engine_type'].apply(standardize_engine_type)
    most_common_engine_type = data['Engine_type'].mode()[0]
    data['Engine_type'] = data['Engine_type'].fillna(most_common_engine_type)
    return data

def handle_capacity_engine_column(data, use_median=False, neutral_value=None):
    data['capacity_Engine'] = data['capacity_Engine'].replace(',', '', regex=True).astype(pd.Int64Dtype())
    median_capacity = data['capacity_Engine'].median()
    mean_capacity = data['capacity_Engine'].mean()
    std_capacity = data['capacity_Engine'].std()
    def adjust_engine_capacity(row):
        if row['Engine_type'] == 'חשמלי':
            if mean_capacity - std_capacity <= row['capacity_Engine'] <= mean_capacity + std_capacity:
                return row['capacity_Engine']
            else:
                return neutral_value if neutral_value is not None else (mean_capacity if not use_median else median_capacity)
        if pd.isna(row['capacity_Engine']):
            return mean_capacity if not use_median else median_capacity
        if row['capacity_Engine'] > 10000:
            return row['capacity_Engine'] / 10
        if row['capacity_Engine'] < 26:
            return row['capacity_Engine'] * 100
        if 0 <= row['capacity_Engine'] < 10:
            return mean_capacity if not use_median else median_capacity
        if 26 < row['capacity_Engine'] < 800:
            return mean_capacity if not use_median else median_capacity
        return row['capacity_Engine']
    data['capacity_Engine'] = data.apply(adjust_engine_capacity, axis=1)
    data['capacity_Engine'] = data['capacity_Engine'].round(0)
    return data

def assign_ownership_value(row):
    ownership_terms_regex = r'ליסינג|השכרה|חברה'
    columns_to_check = ['Prev_ownership', 'Curr_ownership']
    for col in columns_to_check:
        if pd.notna(row[col]) and re.search(ownership_terms_regex, row[col]):
            return 1
    try:
        if pd.notna(row['Description']) and re.search(r'\bליסינג\b', row['Description']):
            return 1
    except KeyError:
        pass
    return 0

def handle_area_column(data):
    hebrew_areas = {
        r"\bגליל(?:\s+ועמקים)?\b": "גליל ועמקים",
        r"\ב חיפ(?:ה\s+וחוף\s+הכרמל)?\b": "חיפה וחוף הכרמל",
        r"\ב עמק\s+יזרעאל\b": "עמק יזרעאל",
        r"\ב קריות\b": "קריות",
        r"\ב טבריה\s+והסביבה\b": "טבריה והסביבה",
        r"\ב עכו\s+-\s+נהריה\b": "עכו - נהריה",
        r"\ב כרמיאל\s+והסביבה\b": "כרמיאל והסביבה",
        r"\ב מושבים\s+בצפון\b": "מושבים בצפון",
        r"\ב רעננה\s+-\s+כפר\s+סבא\b": "רעננה - כפר סבא",
        r"\ב נתניה\s+והסביבה\b": "נתניה והסביבה",
        r"\ב רמת\s+השרון\s+-\s+הרצליה\b": "רמת השרון - הרצליה",
        r"\ב מושבים\s+בשרון\b": "מושבים בשרון",
        r"\ב חדרה\s+ותושבי\s+עמק\s+חפר\b": "חדרה ותושבי עמק חפר",
        r"\ב פרדס\s+חנה\s+-\s+כרכור\b": "פרדס חנה - כרכור",
        r"\ב הוד\s+השרון\s+והסביבה\b": "הוד השרון והסביבה",
        r"\ב יישובי\s+השומרון\b": "יישובי השומרון",
        r"\ב זכרון\s+-\s+בנימינה\b": "זכרון - בנימינה",
        r"\ב אזור\s+השרון\s+והסביבה\b": "אזור השרון והסביבה",
        r"\ב תל\s+אביב\b": "תל אביב",
        r"\ב חולון\s+-\s+בת\s+ים\b": "חולון - בת ים",
        r"\ב ראשל\"צ\s+והסביבה\b": "ראשל\"צ והסביבה",
        r"\ב רמת\s+גן\s+-\s+גבעתיים\b": "רמת גן - גבעתיים",
        r"\ב פתח\s+תקווה\s+והסביבה\b": "פתח תקווה והסביבה",
        r"\ב ראש\s+העין\s+והסביבה\b": "ראש העין והסביבה",
        r"\ב בני\s+ברק\b": "בני ברק",
        r"\ב ישובים\s+במרכז\b": "ישובים במרכז",
        r"\ב ירושלים\s+והסביבה\b": "ירושלים והסביבה",
        r"\ב מודיעין\s+והסביבה\b": "מודיעין והסביבה",
        r"\ב מושבים\s+באזור\s+ירושלים\b": "מושבים באזור ירושלים",
        r"\ב אשדוד\s+-\s+אשקלון\b": "אשדוד - אשקלון",
        r"\ב נס\s+ציונה\s+-\s+רחובות\b": "נס ציונה - רחובות",
        r"\ב גדרה\s+יבנה\s+והסביבה\b": "גדרה יבנה והסביבה",
        r"\ב רמלה לוד\b": "רמלה לוד",
        r"\ב מושבים\s+בשפלה\b": "מושבים בשפלה",
        r"\ב באר\s+שבע\s+והסביבה\b": "באר שבע והסביבה",
        r"\ב אילת\s+והערבה\b": "אילת והערבה",
        r"\ב מושבים\s+בדרום\b": "מושבים בדרום"
    }
    for pattern, replacement in hebrew_areas.items():
        data["Area"] = data["Area"].str.replace(pattern, replacement, regex=True)
    region_mappings = {
        "אזור צפון": [
            "גליל ועמקים",
            "חיפה וחוף הכרמל",
            "עמק יזרעאל",
            "קריות",
            "טבריה והסביבה",
            "עכו - נהריה",
            "כרמיאל והסביבה",
            "מושבים בצפון"
        ],
        "אזור השרון והסביבה": [
            "רעננה - כפר סבא",
            "נתניה והסביבה",
            "רמת השרון - הרצליה",
            "מושבים בשרון",
            "חדרה ותושבי עמק חפר",
            "פרדס חנה - כרכור",
            "הוד השרון והסביבה",
            "זכרון - בנימינה",
            "אזור השרון והסביבה"
        ],
        "אזור מרכז": [
            "תל אביב",
            "חולון - בת ים",
            "ראשל\"צ והסביבה",
            "רמת גן - גבעתיים",
            "פתח תקווה והסביבה",
            "ראש העין והסביבה",
            "בני ברק",
            "ישובים במרכז"
        ],
        "אזור ירושלים והסביבה": [
            "ירושלים והסביבה",
            "מודיעין והסביבה",
            "מושבים באזור ירושלים"
        ],
        "אזור השפלה והסביבה": [
            "אשדוד - אשקלון",
            "נס ציונה - רחובות",
            "גדרה יבנה והסביבה",
            "רמלה לוד",
            "מושבים בשפלה"
        ],
        "אזור דרום": [
            "באר שבע והסביבה",
            "אילת והערבה",
            "מושבים בדרום"
        ]
    }
    data["Area_new"] = None
    for region, areas in region_mappings.items():
        data.loc[data["Area"].isin(areas), "Area_new"] = region
    return data

def fill_missing_area(data):
    value_counts = data['Area_new'].value_counts(normalize=True)
    categories = value_counts.index.tolist()
    probabilities = value_counts.values.tolist()
    def fill_with_probability():
        return np.random.choice(categories, p=probabilities)
    data['Area_new'] = data['Area_new'].apply(lambda x: fill_with_probability() if pd.isna(x) else x)
    return data

def clean_km_column(data):
    data['Km'] = data['Km'].astype(str).replace(',', '', regex=True).str.strip()
    data['Km'] = data['Km'].str.replace(r'\D+', '', regex=True)
    data['Km'] = pd.to_numeric(data['Km'], errors='coerce').fillna(0)
    data['Km'] = data['Km'].round().astype(int)
    return data

def handle_km_column(data, use_median=True):
    data = clean_km_column(data)
    current_year = data['Year'].max()
    data['processed'] = False
    for idx, row in data.iterrows():
        if row['Km'] <= 9 and row['Year'] < current_year and not row['processed']:
            data.at[idx, 'Km'] = (current_year - row['Year']) * 16700
            data.at[idx, 'processed'] = True
            continue
        if 10 <= row['Km'] < 1000 and not row['processed']:
            temp_km = row['Km'] * 1000
            vehicle_age = current_year - row['Year']
            std_dev = data['Km'].std()
            lower_limit = (vehicle_age - 0.5) * 16700 - (0.5 * std_dev)
            upper_limit = (vehicle_age + 0.5) * 16700 + (0.5 * std_dev)
            if lower_limit <= temp_km <= upper_limit:
                data.at[idx, 'Km'] = temp_km
            else:
                data.at[idx, 'Km'] = (lower_limit + upper_limit) / 2
            data.at[idx, 'processed'] = True
            continue
        if row['Year'] < current_year - 1 and 1000 <= row['Km'] < 10000 and not row['processed']:
            data.at[idx, 'Km'] = row['Km'] * 10
            data.at[idx, 'processed'] = True
            continue
        if row['Year'] in [current_year, current_year - 1] and row['Hand'] > 1 and row['Km'] < 100000 and not row['processed']:
            data.at[idx, 'Km'] = row['Km'] * 10
            data.at[idx, 'processed'] = True
            continue
    data.drop(columns=['processed'], inplace=True)
    if use_median:
        fill_value = data['Km'].median()
    else:
        fill_value = data['Km'].mean()
    data['Km'] = data['Km'].fillna(fill_value)
    Q1 = data['Km'].quantile(0.25)
    Q3 = data['Km'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data.loc[data['Km'] > upper_bound, 'Km'] = upper_bound
    data.loc[data['Km'] < lower_bound, 'Km'] = lower_bound
    return data


def prepare_data(data):
    data['Hand'] = pd.to_numeric(data['Hand'], errors='coerce')
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data['capacity_Engine'] = pd.to_numeric(data['capacity_Engine'], errors='coerce')
    data['Km'] = pd.to_numeric(data['Km'], errors='coerce')
    try:
        data['Test'] = pd.to_numeric(data['Test'], errors='coerce')
    except KeyError:
        pass
    data = remove_Duplicates(data)
    data['manufactor'] = data['manufactor'].apply(standardize_manufactor)
    data = remove_manufactor_name_in_model_col(data)
    data['model'] = data['model'].apply(clean_model_name)
    data['model'] = data['model'].apply(normalize_model_name)
    data = handle_hand_column(data)
    data = handle_gear_column(data)
    data = handle_engine_type_column(data)
    data = handle_capacity_engine_column(data)
    data['Ownership_Value'] = data.apply(assign_ownership_value, axis=1).astype('category')
    data = handle_area_column(data)
    data = fill_missing_area(data)
    data = handle_km_column(data)
    data = bin_year_column(data, num_bins=5)
    columns_to_drop = ['Prev_ownership','Curr_ownership','Area','City','Pic_num','Cre_date','Repub_date','Description','Test','Supply_score','Color','Gear']
    for column in columns_to_drop:
        try:
            data = data.drop(columns=[column])
        except KeyError:
            pass
    return data
