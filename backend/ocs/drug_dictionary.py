"""
주요 약물 한영 성분명 사전
DDInter 매칭을 위한 핵심 약물 매핑
"""

# 한글 → 영문 성분명 매핑
KOREAN_TO_ENGLISH_DRUG_DICT = {
    # 테스트 케이스 약물들
    "시프로플록사신": "ciprofloxacin",
    "테오필린": "theophylline",
    "아지스로마이신": "azithromycin",
    "하이드록시클로로퀸": "hydroxychloroquine",
    "독시사이클린": "doxycycline",
    "아목시실린": "amoxicillin",
    "이부프로펜": "ibuprofen",
    "오메프라졸": "omeprazole",
    "메트로니다졸": "metronidazole",
    "에탄올": "ethanol",
    "알코올": "ethanol",
    
    # 당뇨
    "메트포르민": "metformin",
    "글리메피라이드": "glimepiride",
    "인슐린": "insulin",
    "프레드니솔론": "prednisolone",
    "글리벤클라미드": "glyburide",
    "플루코나졸": "fluconazole",
    
    # 정신과
    "플루옥세틴": "fluoxetine",
    "트라마돌": "tramadol",
    "리튬": "lithium",
    "히드로클로로티아지드": "hydrochlorothiazide",
    "미르타자핀": "mirtazapine",
    "알프라졸람": "alprazolam",
    "에스시탈로프람": "escitalopram",
    "할로페리돌": "haloperidol",
    "클래리스로마이신": "clarithromycin",
    
    # 심혈관
    "암로디핀": "amlodipine",
    "심바스타틴": "simvastatin",
    "딜티아젬": "diltiazem",
    "프로프라놀롤": "propranolol",
    "디곡신": "digoxin",
    "스피로놀락톤": "spironolactone",
    "푸로세미드": "furosemide",
    "로사르탄": "losartan",
    "아스피린": "aspirin",
    "클로피도그렐": "clopidogrel",
    
    # 복합
    "레보티록신": "levothyroxine",
    "칼슘": "calcium",
    "철분": "iron",
    "와파린": "warfarin",
    "아미오다론": "amiodarone",
    "아토르바스타틴": "atorvastatin",
    "타크로리무스": "tacrolimus",
    "타목시펜": "tamoxifen",
    "타목시펜시트르산염": "tamoxifen",
    "와파린": "warfarin",
    "와파린나트륨": "warfarin",
    
    # 추가 주요 약물
    "이트라코나졸": "itraconazole",
    "아세트아미노펜": "acetaminophen",
    "파라세타몰": "acetaminophen",
    "타이레놀": "acetaminophen",
    "나프록센": "naproxen",
    "셀레콕시브": "celecoxib",
    "디클로페낙": "diclofenac",
    "케토코나졸": "ketoconazole",
    "에리스로마이신": "erythromycin",
    "시클로스포린": "cyclosporine",
    "펜토인": "phenytoin",
    "카르바마제핀": "carbamazepine",
    "발프로산": "valproic acid",
    "세르트랄린": "sertraline",
    "파록세틴": "paroxetine",
    "벤라팍신": "venlafaxine",
    "디아제팜": "diazepam",
    "로라제팜": "lorazepam",
    "졸피뎀": "zolpidem",
    "리스페리돈": "risperidone",
    "올란자핀": "olanzapine",
    "퀘티아핀": "quetiapine",
    "라모트리진": "lamotrigine",
    "가바펜틴": "gabapentin",
    "프레가발린": "pregabalin",
    "코데인": "codeine",
    "펜타닐": "fentanyl",
    "모르핀": "morphine",
    "옥시코돈": "oxycodone",
    
    # --- 30종 테스트 케이스 추가 (브랜드명 포함) ---
    # 해열진통제
    "타이레놀": "acetaminophen",
    "펜잘": "acetaminophen", # 복합제지만 주성분 매핑
    "게보린": "acetaminophen", # 복합제지만 주성분 매핑
    "애드빌": "ibuprofen",
    "부루펜": "ibuprofen",
    "덱시부프로펜": "dexibuprofen",
    "판피린": "acetaminophen", # 판피린티
    "콜대원": "acetaminophen", # 복합제
    "로키논": "loxoprofen",
    "쿠멘": "diclofenac", # 추정 (또는 유사 계열)
    "싸이클로날": "acetaminophen", # 추정 (혹은 확인 필요)
    
    # 항생제/항진균제
    "스포라녹스": "itraconazole",
    "에리트로마이신": "erythromycin",
    "클래리스로마이신": "clarithromycin",
    "크라비트": "levofloxacin",
    "플래그": "metronidazole", # Flagyl
    "후라시닐": "metronidazole",
    
    # 심혈관 (고혈압, 고지혈증, 항응고)
    "리피토": "atorvastatin",
    "아토르바스타틴": "atorvastatin",
    "노바스크": "amlodipine",
    "디오반": "valsartan",
    "크레스토": "rosuvastatin",
    "로수바스타틴": "rosuvastatin",
    "테논민": "atenolol",
    "아테놀롤": "atenolol",
    "딜티아젬": "diltiazem",
    "프라비딘": "clopidogrel", # 클로피도그렐 성분
    "아스피린프로텍트": "aspirin",
    
    # 당뇨/대사
    "다이아벡스": "metformin",
    "글루파": "metformin",
    "아마릴": "glimepiride",
    "자누비아": "sitagliptin",
    "다오닐": "glyburide",
    "신티로이드": "levothyroxine",
    "아카보스": "acarbose",
    
    # 소화기
    "가스모틴": "mosapride",
    "훼스탈": "pancreatin", # 소화효소
    "오메프라졸": "omeprazole",
    
    # 호흡기/알레르기
    "지르텍": "cetirizine",
    "클라리틴": "loratadine",
    "무코펙트": "ambroxol",
    "유니필": "theophylline",
    "코푸": "dihydrocodeine", # 코푸시럽
    
    # 정신과/신경과
    "프로작": "fluoxetine",
    "자낙스": "alprazolam",
    "스틸녹스": "zolpidem",
    "렉사프로": "escitalopram",
    "세로켈": "quetiapine",
    "테그레톨": "carbamazepine",
    "리단": "lithium", # 리단정
    
    # 기타
    "머시론": "desogestrel", # 피임약 (복합)
    "조영제": "contrast media", # 일반 명칭 매핑 (특정 성분은 아님)
    "이오헥솔": "iohexol", # 대표적 조영제
    "술": "ethanol",
    "소주": "ethanol",
    "자몽": "grapefruit",
    "자몽주스": "grapefruit juice",
    "칼륨": "potassium",
    
    # --- Missing Ingredients Update (Top Frequency) ---
    "록소프로펜": "loxoprofen",
    "멜록시캄": "meloxicam",
    "세프트리악손": "ceftriaxone",
    "레보드로프로피진": "levodropropizine",
    "클래리트로마이신": "clarithromycin",
    "케토프로펜": "ketoprofen",
    "세티리진": "cetirizine",
    "레바미피드": "rebamipide",
    "몬테루카스트": "montelukast",
    "카르베딜롤": "carvedilol",
    "록시트로마이신": "roxithromycin",
    "에르도스테인": "erdosteine",
    "발사르탄": "valsartan",
    "아세틸시스테인": "acetylcysteine",
    "에페리손": "eperisone",
    "쿠에티아핀": "quetiapine",
    "토피라메이트": "topiramate",
    "탐스로신": "tamsulosin",
    "니자티딘": "nizatidine",
    "돔페리돈": "domperidone",
    "플루르비프로펜": "flurbiprofen",
    "란소프라졸": "lansoprazole",
    "암브록솔": "ambroxol",
    "테라조신": "terazosin",
    "이토프리드": "itopride",
    "티옥트": "thioctic acid", # 티옥트산
    "리세드론": "risedronate",
    "세픽심": "cefixime",
    "케토티펜": "ketotifen",
    "라베프라졸": "rabeprazole",
    "모사프리드": "mosapride",
    "세파클러": "cefaclor",
    "메틸프레드니솔론": "methylprednisolone",
    "아테놀롤": "atenolol",
    "실니디핀": "cilnidipine",
    "피타바스타틴": "pitavastatin",
    "오로트산": "orotic acid",
    "카드뮴": "cadmium", # unlikely drug but in namings
    "우르소데옥시콜": "ursodeoxycholic acid",
    "실데나필": "sildenafil",
    "타다라필": "tadalafil",
    "레보플록사신": "levofloxacin",
    "세프포독심": "cefpodoxime",
    "텔미사르탄": "telmisartan",
    "칸데사르탄": "candesartan",
    "이르베사르탄": "irbesartan",
    "올메사르탄": "olmesartan",
    
    # 4차: 암 환자 주요 약물 및 상호작용 약물 (항암제, 항생제, 항진균제, 항고혈압제 등)
    # 항암제 (Cytotoxic)
    "독소루비신": "doxorubicin",
    "에피루비신": "epirubicin",
    "파클리탁셀": "paclitaxel",
    "도세탁셀": "docetaxel",
    "시클로포스파미드": "cyclophosphamide",
    "플루오로우라실": "fluorouracil",
    "카페시타빈": "capecitabine",
    "옥살리플라틴": "oxaliplatin",
    "시스플라틴": "cisplatin",
    "카보플라틴": "carboplatin",
    "젬시타빈": "gemcitabine",
    "이리노테칸": "irinotecan",
    "에토포시드": "etoposide",
    "메토트렉세이트": "methotrexate",
    "이포스파미드": "ifosfamide",
    "빈크리스틴": "vincristine",
    "빈블라스틴": "vinblastine",
    
    # 표적/호르몬 치료제
    "트라스투주맙": "trastuzumab",
    "이마티닙": "imatinib",
    "게피티닙": "gefitinib",
    "엘로티닙": "erlotinib",
    "크리조티닙": "crizotinib",
    "아나스트로졸": "anastrozole",
    "레트로졸": "letrozole",
    "엑스메스탄": "exemestane",
    "고세렐린": "goserelin",
    "류프로렐린": "leuprorelin",
    "풀베스트란트": "fulvestrant",
    "팔보시클립": "palbociclib",
    "올라파립": "olaparib",

    # 면역항암제
    "펨브롤리주맙": "pembrolizumab",
    "니볼루맙": "nivolumab",
    "아테졸리주맙": "atezolizumab",

    # 보조 요법 및 주요 상호작용 약물
    "온단세트론": "ondansetron",
    "팔로노세트론": "palonosetron",
    "아프레피탄트": "aprepitant",
    "덱사메타손": "dexamethasone",
    "프레드니솔론": "prednisolone",
    "케토코나졸": "ketoconazole",
    "이트라코나졸": "itraconazole",
    "플루코나졸": "fluconazole",
    "보리코나졸": "voriconazole",
    "포사코나졸": "posaconazole",
    "에리트로마이신": "erythromycin",
    "클래리스로마이신": "clarithromycin",
    "카르바마제핀": "carbamazepine",
    "페니토인": "phenytoin",
    "페노바르비탈": "phenobarbital",
    "리팜핀": "rifampin",
    "아미오다론": "amiodarone",
    "드론다론": "dronedarone",
    "베라파밀": "verapamil",
    "딜티아젬": "diltiazem",
    "디곡신": "digoxin",
    "와파린": "warfarin",
    "아픽사반": "apixaban",
    "에독사반": "edoxaban",
    "다비가트란": "dabigatran",
    "리바록사반": "rivaroxaban",
    "티카그렐러": "ticagrelor",
    
    # 3차 대규모 추가 (Unmapped 상위 항목)
    "시메티딘": "cimetidine",
    "이소트레티노인": "isotretinoin",
    "레비티라세탐": "levetiracetam",
    "리마프로스트": "limaprost",
    "리마프로스트알파덱스": "limaprost",
    "알마게이트": "almagate",
    "오메가-3-에틸에스테르90": "omega-3-acid ethyl esters",
    "오메가-3": "omega-3",
    "메로페넴": "meropenem",
    "카르복시메틸셀룰로오스": "carboxymethylcellulose",
    "피록시캄": "piroxicam",
    "디오스민": "diosmin",
    "슈가마덱스": "sugammadex",
    "탈니플루메이트": "talniflumate",
    "도베실": "dobesilate",
    "도베실산": "dobesilate",
    "아세브로필린": "acebrophylline",
    "클로닉신": "clonixin",
    "클로닉신리시네이트": "clonixin lysinate",
    "클로르헥시딘": "chlorhexidine",
    "폴리데옥시리보뉴클레오티드": "polydeoxyribonucleotide",
    "칼리디노게나제": "kallidinogenase",
    "세프포독심프록세틸": "cefpodoxime proxetil",
    "메트포르민": "metformin",
    "리나글립틴": "linagliptin",

    # 2차 대규모 추가 (전수 조사 기반)
    "로수바스타틴": "rosuvastatin",
    "도네페질": "donepezil",
    "아토르바스타틴": "atorvastatin",
    "시타글립틴": "sitagliptin",
    "은행엽": "ginkgo leaf",
    "은행엽건조엑스": "ginkgo biloba",
    "에스오메프라졸": "esomeprazole",
    "히알루론산": "hyaluronic acid",
    "히알루론": "hyaluronic acid",
    "피나스테리드": "finasteride",
    "오셀타미비르": "oseltamivir",
    "세레콕시브": "celecoxib",
    "리바록사반": "rivaroxaban",
    "아세클로페낙": "aceclofenac",
    "솔리페나신": "solifenacin",
    "글리메피리드": "glimepiride",
    "테르비나핀": "terbinafine",
    "콜린알포세레이트": "choline alfoscerate",
    "사르포그렐레이트": "sarpogrelate",
    "리바스티그민": "rivastigmine",
    "실로스타졸": "cilostazol",
    "피오글리타존": "pioglitazone",
    "엠파글리플로진": "empagliflozin",
    "레보세티리진": "levocetirizine",
    "팜시클로비르": "famciclovir",
    "메만틴": "memantine",
    "베포타스틴": "bepotastine",
    "다파글리플로진": "dapagliflozin",
    "오플록사신": "ofloxacin",
    "파모티딘": "famotidine",
    "올로파타딘": "olopatadine",
    "두타스테리드": "dutasteride",
    "에피나스틴": "epinastine",
    "트리메부틴": "trimebutine",
    "티로프라미드": "tiropramide",
    "리도카인": "lidocaine",
    "모메타손": "mometasone",
    "모메타손푸로에이트": "mometasone",
    "리나글립틴": "linagliptin",
    "아시클로버": "acyclovir",
    "라니티딘": "ranitidine",
    "레보설피리드": "levosulpiride",
    "엔테카비르": "entecavir",
    "에제티미브": "ezetimibe",
    "이반드론산": "ibandronate",
    "이반드론": "ibandronate",
    "밀크시슬": "milk thistle",
    "밀크시슬열매건조엑스": "milk thistle",
    "아픽사반": "apixaban",
    "알티옥트": "thioctic acid",
    "알티옥트트로메타민": "thioctic acid"
}


# 영문 소문자화 및 역방향 매핑
KOREAN_TO_ENGLISH = {k.lower(): v.lower() for k, v in KOREAN_TO_ENGLISH_DRUG_DICT.items()}
ENGLISH_TO_KOREAN = {v.lower(): k.lower() for k, v in KOREAN_TO_ENGLISH_DRUG_DICT.items()}

def get_english_name(korean_name: str) -> str:
    """한글 성분명 → 영문 성분명"""
    name = korean_name.lower().strip()
    
    # 1. Exact match
    res = KOREAN_TO_ENGLISH.get(name, None)
    if res: return res

    # 2. Contains match (Longest key first)
    # e.g. "국제시메티딘" contains "시메티딘" -> return "cimetidine"
    # Sort keys by length descending to match specific first
    keys = sorted(KOREAN_TO_ENGLISH.keys(), key=len, reverse=True)
    for k in keys:
        if len(k) < 2: continue
        if k in name:
            return KOREAN_TO_ENGLISH[k]
            
    return None

def get_korean_name(english_name: str) -> str:
    """영문 성분명 → 한글 성분명"""
    return ENGLISH_TO_KOREAN.get(english_name.lower(), None)

def translate_ingredient(name: str) -> list:
    """
    성분명 번역 (양방향)
    
    Returns:
        list: [원본, 번역본] 또는 [원본]
    """
    result = [name.lower()]
    
    # 한글인지 영문인지 판단
    if any(ord(c) > 127 for c in name):  # 한글 포함
        eng = get_english_name(name)
        if eng:
            result.append(eng)
    else:  # 영문
        kor = get_korean_name(name)
        if kor:
            result.append(kor)
    
    return result
