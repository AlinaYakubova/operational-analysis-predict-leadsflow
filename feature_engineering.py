import pandas as pd
import config

SOURCE_ENCODING = {'paid_search': 0, 'paid_social': 1, 'organic': 2, 'unknown': 3}
PIPELINE_ENCODING = {'CIS': 0, 'Asia': 1, 'GB': 2, 'LatAm': 3, 'Turkey': 4, 'EU': 5, 'Brazil': 6, 'Other': 7}
COURSE_ENCODING = {'General': 0, 'Coding': 1, 'Math': 2, 'Media': 3, 'other': 4}

SOURCE_GROUPS = {
    'paid_search': {
        'google', 'google_search', 'google_gdn', 'google_pmax',
        'google_youtube', 'google_demand_gen', 'google_android', 'google_ios',
        'yandex', 'yandex_direct',
    },
    'paid_social': {
        'facebook', 'facebook_page', 'instagram_page', 'vk', 'vk_ads', 'vk ads',
        'vk_mp', 'vk_leadform', 'vk_page', 'tiktok', 'tiktok_page', 'influence',
    },
    'organic': {
        'referral', 'family', 'website', 'blog', 'email', 'telegram',
        'telegram_page', 'youtube', 'youtube_page', 'social', 'dzen_page',
        'webinar', 'webflow', 'platform', 'kodland_app', 'chatra',
    },
}

SOURCE_MAP = {
    value: group
    for group, values in SOURCE_GROUPS.items()
    for value in values
}

PIPELINE_MAP = {
    'Онлайн Воронка': 'CIS',
    'Asia | Indonesia': 'Asia',
    'Asia | Malaysia': 'Asia',
    'English Pipeline': 'GB',
    'LatAm | Peru, Mexico, Colombia': 'LatAm',
    'MENAP | Turkey': 'Turkey',
    'EU | Italy': 'EU',
    'GCC Pipeline': 'Other',
    'LatAm | Spain': 'EU',
    'EU | Poland': 'EU',
    'LatAm | Brazil': 'Brazil',
    'Supply': 'Other',
}

FEATURE_SPECS = {
    'source': {
        'feature': 'source_group',
        'mapping': SOURCE_MAP,
        'encoding': SOURCE_ENCODING,
        'default': 'unknown',
        'lowercase': True,
    },
    'general_amocrm_pipeline': {
        'feature': 'pipeline_group',
        'mapping': PIPELINE_MAP,
        'encoding': PIPELINE_ENCODING,
        'default': 'Other',
        'lowercase': False,
    },
    'marketing_course_global': {
        'feature': 'course_group',
        'mapping': {course: course for course in COURSE_ENCODING if course != 'other'},
        'encoding': COURSE_ENCODING,
        'default': 'other',
        'lowercase': False,
    },
}

def map_values(series, mapping, default, lowercase=False):
    values = series.astype('string').str.strip()
    if lowercase:
        values = values.str.lower()
    return values.map(mapping).fillna(default)

def first_mode(series):
    modes = series.mode()
    if modes.empty:
        return pd.NA
    return modes.iloc[0]

def build_weekly_features(raw):
    for raw_col, spec in FEATURE_SPECS.items():
        raw[spec['feature']] = map_values(
            raw[raw_col],
            mapping=spec['mapping'],
            default=spec['default'],
            lowercase=spec['lowercase'],
        ).map(spec['encoding'])

    feature_cols = [spec['feature'] for spec in FEATURE_SPECS.values()]
    return raw.groupby('week')[feature_cols].agg(first_mode).sort_index()

def main():
    print("=== RUNNING FEATURE ENGINEERING ===")

    raw = pd.read_csv(config.WEEKLY_DATA_PATH)
    raw['week'] = pd.to_datetime(raw['week'])

    new_features = build_weekly_features(raw)

    processed = pd.read_csv("processed_weekly_global.csv", index_col=0, parse_dates=True)
    enriched = processed.join(new_features, how='left')
    enriched[new_features.columns] = enriched[new_features.columns].fillna(0)

    enriched.to_csv("processed_weekly_enriched.csv")
    print(f"Saved processed_weekly_enriched.csv - {enriched.shape[0]} rows, {enriched.shape[1]} columns")
    print("New features:", list(new_features.columns))


if __name__ == "__main__":
    main()
