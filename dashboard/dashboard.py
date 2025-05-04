import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
import re
import emoji
import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
import joblib
sns.set(style='dark')


# ------------------------------------------------------- Class Analisis Sentimen -------------------------------------------------------
# Pastikan stopwords sudah tersedia
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# List stopwords
listStopwords = set(stopwords.words('indonesian') + stopwords.words('english'))
listStopwords.update([
    'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya',
    'gaa', 'loh', 'kah', 'woi', 'woii', 'woy', 'anjay', 'gk', 'g'
])

# Slang words dictionary
slangwords = {"@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal",
              "tpi":"tapi","tp":"tapi","ktolong":"ka tolong","g":"tidak","anjiiiing":"umpat","knp":"kenapa","tibatiba":"tiba-tiba","ad":"ada",
              "tbtb":"tiba-tiba","yt":"youtube","ig":"instagram","gk":"tidak","yg":"yang","moga":"semoga","pake":"pakai","ngirim":"kirim",
              "muas":"puas","sdh":"sudah","lg":"lagi","sya":"saya","klo":"kalau","knpa":"kenapa","tdk":"tidak","sampe":"sampai","kayak":"seperti",
              "cuman":"hanya","prose":"proses","ny":"","jd":"jadi","dgn":"dengan","jg":"juga","tf":"transfer","sampe":"sampai","ngirim":"kirim",
              "bagu":"bagus","skrg":"sekarang","nunggu":"tunggu","udah":"sudah","uda":"sudah","pk":"pakai","@": "di", "abis": "habis", "wtb": "beli",
              "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia",
              "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah",
              "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku",
              "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa",
              "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik",
              "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang",
              "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur",
              "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar",
              "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana",
              "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar",
              "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng",
              "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur",
              "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi",
              "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja",
              "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci",
              "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor",
              "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan",
              "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan",
              "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian",
              "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile",
              "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan",
              "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi",
              "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda",
              "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo",
              "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh",
              "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi",
              "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif ", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus",
              "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah",
              "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba",
              "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras",
              "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual",
              "tioghoa": "tionghoa", "faktakta": "fakta", "blue":"blu", "rb":"ribu", "lemot": "lelet", "sohib": "teman", "eror":"error", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin"}


class SentimentAnalyzer:
    def __init__(self, tfidf_vectorizer: TfidfVectorizer, model: BaseEstimator):
        self.vectorizer = tfidf_vectorizer
        self.model = model

    def cleaning_text(self, text: str) -> str:
        text = str(text).lower()
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\b(?:\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?)?\d{4,}\b', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
        text = text.replace('_', ' ')
        text = text.replace('-', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def normalisasi_slang(self, text: str) -> str:
        hasil = []
        for kata in text.split():
            kata_baru = slangwords.get(kata.lower(), kata)
            hasil.append(kata_baru)
        return ' '.join(hasil)

    def tokenizing_text(self, text: str) -> list:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in listStopwords]
        return tokens

    def proses_teks_input(self, text: str, show_log: bool = False) -> str:
        if show_log: print(f"Teks asli: {text}")
        
        # Tahap 1: Preprocessing
        teks = self.cleaning_text(text)
        teks = self.normalisasi_slang(teks)
        tokens = self.tokenizing_text(teks)
        teks = ' '.join(tokens)
        
        if show_log: print(f"Teks praproses: {teks}")

        # Tahap 2: TF-IDF transform
        vector = self.vectorizer.transform([teks]).toarray()

        # Tahap 3: Prediksi
        pred = self.model.predict(vector)[0]

        return f"Kalimat terdeteksi {pred}"

# ------------------------------------------------------- Function -------------------------------------------------------
# Fungsi untuk load dataset
def load_data(path):
    return pd.read_csv(path)

# Fungsi untuk menampilkan pie chart distribusi analisis sentimen
def pie_sentimen(data_path): 
    if os.path.exists(data_path):
        # Load dataset
        data_model = pd.read_csv(data_path)

        # Hitung jumlah masing-masing sentimen
        polarity_counts = data_model['polarity'].value_counts()

        # Pie chart
        fig, ax = plt.subplots(figsize=(4, 4))  # Ukuran lebih proporsional
        wedges, texts, autotexts = ax.pie(
            polarity_counts,
            labels=polarity_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=['lightblue', 'lightcoral', 'lightgreen'],
            textprops={'fontsize': 10}
        )
        ax.axis('equal')  # Agar lingkaran tetap bulat

        # Tampilkan plot di Streamlit
        st.pyplot(fig)
    else:
        st.error(f"File tidak ditemukan: {data_path}")
# Fungsi untuk menampilkan dsitribusi rating sentimen
def rating_sentimen(data_path):
    if os.path.exists(data_path):
        data_model = pd.read_csv(data_path)

        # Hitung jumlah sentimen per score
        sentiment_by_score = data_model.groupby(['score', 'sentiment']) \
                                       .size() \
                                       .unstack(fill_value=0)

        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_by_score.plot(kind='bar',
                                stacked=True,
                                ax=ax,
                                color=['lightgreen', 'lightblue', 'lightcoral'])  # Positif, Netral, Negatif

        ax.set_title('Distribusi Sentimen per Rating')
        ax.set_xlabel('Rating (score)')
        ax.set_ylabel('Jumlah Ulasan')
        ax.set_xticks(range(len(sentiment_by_score.index)))
        ax.set_xticklabels(sentiment_by_score.index, rotation=0)
        ax.legend(title='Sentiment')

        plt.tight_layout()
        st.pyplot(fig)

        return sentiment_by_score  # Mengembalikan hasil pivot tabel untuk tab tabel
    else:
        st.error(f"File tidak ditemukan: {data_path}")
        return None

# Fungsi untuk menampilkan wordcloud
def tampilkan_wordcloud(data_path, sentiment_value, label=""):
    if os.path.exists(data_path):
        data_model = pd.read_csv(data_path)

        st.subheader(f"WordCloud Sentimen {label}")

        teks = " ".join(data_model[data_model['sentiment'] == sentiment_value]['text_akhir'].dropna())

        if not teks.strip():
            st.warning(f"Tidak ada ulasan dengan sentimen {label.lower()}.")
            return

        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color='white',
            min_font_size=10,
            prefer_horizontal=1.0
        ).generate(teks)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.error(f"File tidak ditemukan: {data_path}")

# Halaman sidebar
with st.sidebar:
    st.image("../image/blu.png")  # Ganti path sesuai lokasi logo kamu

    # Navigasi di sidebar
    page = st.selectbox("Navigasi", ["Tentang blu", "Analisis Data Ulasan", "Analisis Sentimen"])

# ------------- PAGE 1: TENTANG BLU -------------
if page == "Tentang blu":
    st.title('Analisis Sentimen Aplikasi blu by BCA Digital :bank:')

    st.header('Apa itu blu by BCA Digital?')
    st.write('''
    blu by BCA Digital adalah aplikasi mobile banking dari BCA Digital yang menawarkan pengalaman perbankan digital yang inovatif dan memudahkan pengelolaan keuangan sehari-hari.
    Dengan desain yang user-friendly, blu menyediakan berbagai fitur untuk memenuhi kebutuhan transaksi finansial pengguna, mulai dari pembukaan rekening hingga transaksi perbankan yang aman dan cepat.
    ''')

    st.header("Fitur Utama Aplikasi blu")
    tab1, tab2, tab3 = st.tabs(["bluSaving", "bluGather", "bluDeposit"])
    with tab1:
        col1, col2 = st.columns([2, 3])  # kolom kiri lebih kecil dari kanan

        with col1:
            st.image("../image/bluSaving.png", use_column_width=True)  # ganti dengan path yang sesuai

        with col2:
            st.subheader("bluSaving")
            st.write("Mau healing, beli gadget, atau nabung dana darurat? semua bisa diwujudin dengan **bluSaving**!")
            st.write("Fitur ini memungkinkan kamu untuk menyimpan uang dalam sampai dengan 20 pos tabungan sesuai kebutuhan. Anda bisa menetapkan target dan waktu untuk masing-masing pos, serta memantau perkembangan tabungan secara real-time.")

    with tab2:
        col1, col2 = st.columns([2, 3])  # kolom kiri lebih kecil dari kanan

        with col1:
            st.image("../image/bluGether.png", use_column_width=True)  # ganti dengan path yang sesuai

        with col2:
            st.subheader("bluGether")
            st.write("""Nabung bareng buat arisan, jalan-jalan, atau patungan makan? **bluGether** solusinya!""")
            st.write("Fitur ini memungkinkan kamu untuk nabung bersama teman-teman mu hingga 100 orang.")

            
    with tab3:
        col1, col2 = st.columns([2, 3])  # kolom kiri lebih kecil dari kanan

        with col1:
            st.image("../image/bluDeposit.png", use_column_width=True)  # ganti dengan path yang sesuai

        with col2:
            st.subheader("bluDeposit")
            st.write("Mau nyimpen dana dengan aman dan tetap mendapatkan imbal hasil? **bluDeposit** solusinya")
            st.write("Fitur deposito berjangka dari blu yang memberikan bunga kompetitif dan pilihan tenor fleksibel.")
            st.write("Pembuatan dan pencairan deposito mulai dari 1 juta, bisa dilakukan langsung dari aplikasi tanpa harus ke bank. Cair sebelum jatuh tempo? Tetap aman, tanpa penalti!")

# ------------- PAGE 2: ANALISIS DATA ULASAN -------------
elif page == "Analisis Data Ulasan":
    st.title("📋 Ulasan Pengguna Aplikasi blu")

    data_path = "../data/data_model.csv"
    data_model = load_data(data_path)

    jumlah_data = st.slider("Tampilkan berapa banyak ulasan?", min_value=5, max_value=50, value=10, step=5)
    rating_filter = st.multiselect("Filter berdasarkan rating:", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    filtered_data = data_model[data_model['score'].isin(rating_filter)].copy()
    filtered_data = filtered_data.head(jumlah_data)
    filtered_data.insert(0, 'No', range(1, len(filtered_data) + 1))
    filtered_data = filtered_data.rename(columns={'content': 'Ulasan', 'score': 'Rating'})

    st.markdown(filtered_data[['No', 'Ulasan', 'Rating']].to_html(index=False, escape=False), unsafe_allow_html=True)

    st.title("📊 Analisis Sentimen Ulasan")

    data_path = "../data/ulasan_sentimen.csv"

    # Distribusi Sentimen (Lexicon-based
    st.header("Distribusi Sentimen (Lexicon-based)")
    col1, col2 = st.columns([2,2])
    with col1: 
        pie_sentimen(data_path)  # pastikan fungsi ini sudah kamu definisikan
    with col2:
        st.write(
            'Dari total 17.533 ulasan, sebanyak 48% bersentimen positif, 35.7% negatif, dan 16.3% netral. Ini menunjukkan mayoritas pengguna puas, namun ulasan negatif masih cukup signifikan.'
        )

    # Distribusi Sentimen per rating
    st.header("Distribusi Sentimen per Rating")
    tab1, tab2 = st.tabs(["Bar Chart", "Tabel"])

    with tab1:
        sentiment_table = rating_sentimen(data_path)  # pastikan fungsi ini tersedia

    with tab2:
        if sentiment_table is not None:
            col1, col2 = st.columns([2, 3])

            with col1:
                st.dataframe(sentiment_table)

            with col2:
                st.markdown("""
                ##### 📝 Penjelasan Tabel Sentimen per Rating

                **Keterangan kolom:**
                - Kolom `-1` → jumlah ulasan **negatif**
                - Kolom `0` → jumlah ulasan **netral**
                - Kolom `1` → jumlah ulasan **positif**

                **Keterangan baris:**
                - Setiap baris merepresentasikan **rating bintang (1-5)** dari pengguna.

                **Insight:**
                - ⭐ **Bintang 5**: Didominasi sentimen **positif**
                - ⭐ **Bintang 1**: Didominasi sentimen **negatif**
                - ⭐ **Bintang 3**: Cenderung **seimbang**
                """, unsafe_allow_html=True)

# WordCloud
    st.header("Visualisasi Kata yang sering muncul")
    tab1, tab2, tab3 = st.tabs(["Positif", "Netral", "Negatif"])

    with tab1:
        tampilkan_wordcloud(data_path, 1, "Positif")

    with tab2:
        tampilkan_wordcloud(data_path, 0, "Netral")

    with tab3:
        tampilkan_wordcloud(data_path, -1, "Negatif")


# ------------- PAGE 3: ANALISIS DATA ULASAN -------------
elif page == "Analisis Sentimen":
    # Load model dan vectorizer
    model = joblib.load("../model/model.joblib")
    vectorizer = joblib.load("../model/tfidf_vectorizer.joblib")

    # Inisialisasi analyzer
    analyzer = SentimentAnalyzer(vectorizer, model)

    st.title("Analisis Sentimen Ulasan")

    kalimat = st.text_input("Masukkan kalimat:")

    if st.button("Input"):
        if kalimat.strip() != "":
            hasil = analyzer.proses_teks_input(kalimat)
            st.success(hasil)
        else:
            st.warning("Kalimat tidak boleh kosong.")