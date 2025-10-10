# 🎬 TikTok Clone API### Step 1: Set Up Your Environment



Backend API cho ứng dụng chia sẻ video kiểu TikTok, được xây dựng với **FastAPI** và **MySQL**.1. **Create a Virtual Environment**:

   ```bash

## 📋 Mục lục   python -m venv venv

   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

- [Tính năng](#-tính-năng)   ```

- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

- [Cài đặt](#-cài-đặt)2. **Install Required Packages**:

- [Cấu hình](#-cấu-hình)   ```bash

- [Chạy dự án](#-chạy-dự-án)   pip install fastapi uvicorn sqlalchemy pymysql

- [API Documentation](#-api-documentation)   ```

- [Cấu trúc Database](#-cấu-trúc-database)

- [API Endpoints](#-api-endpoints)### Step 2: Set Up MySQL Database

- [Testing](#-testing)

1. **Install MySQL**: Make sure you have MySQL installed on your machine.

## ✨ Tính năng

2. **Create a Database**:

### 🔐 **Authentication & Users**   ```sql

- Đăng ký, đăng nhập với JWT token   CREATE DATABASE video_sharing;

- Quản lý profile người dùng   ```

- Upload avatar

- Phân quyền User/Admin3. **Create a User and Grant Privileges**:

   ```sql

### 🎥 **Videos**   CREATE USER 'user'@'localhost' IDENTIFIED BY 'password';

- Upload video với tự động tạo thumbnail   GRANT ALL PRIVILEGES ON video_sharing.* TO 'user'@'localhost';

- CRUD operations cho videos   FLUSH PRIVILEGES;

- Kiểm tra độ dài video (≤ 300 giây)   ```

- Phân loại video: Public/Hidden/Deleted

- HLS streaming support4. **Create a Table for Videos**:

   ```sql

### 💬 **Comments**   USE video_sharing;

- Bình luận trên video

- Soft delete (ẩn/hiện comment)   CREATE TABLE videos (

- Quản lý comment bởi admin       id INT AUTO_INCREMENT PRIMARY KEY,

       title VARCHAR(255) NOT NULL,

### 👥 **Social Features**       description TEXT,

- Like/Unlike videos       url VARCHAR(255) NOT NULL,

- Follow/Unfollow users       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

- Bookmark videos   );

- Thống kê followers, following, likes   ```



### 💌 **Messages**### Step 3: Create FastAPI Application

- Gửi tin nhắn giữa users

- Upload media trong tin nhắn1. **Project Structure**:

- Xem danh sách cuộc hội thoại   ```

- Đánh dấu đã đọc/chưa đọc   video_sharing/

   ├── main.py

### 🚨 **Reports**   ├── models.py

- Báo cáo vi phạm (Video/Comment/User)   ├── database.py

- Admin xử lý báo cáo   └── schemas.py

- Theo dõi trạng thái báo cáo   ```



### 🔔 **Notifications**2. **`database.py`**: Set up the database connection.

- Thông báo realtime   ```python

- Đánh dấu đã đọc   from sqlalchemy import create_engine

- Đếm thông báo chưa đọc   from sqlalchemy.ext.declarative import declarative_base

   from sqlalchemy.orm import sessionmaker

## 🛠 Công nghệ sử dụng

   SQLALCHEMY_DATABASE_URL = "mysql+pymysql://user:password@localhost/video_sharing"

- **FastAPI** 0.104.1 - Web framework

- **SQLAlchemy** 2.0.23 - ORM   engine = create_engine(SQLALCHEMY_DATABASE_URL)

- **MySQL** - Database   SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

- **PyMySQL** - MySQL driver

- **Pydantic** 2.5.0 - Data validation   Base = declarative_base()

- **JWT** (python-jose) - Authentication   ```

- **Passlib** - Password hashing

- **OpenCV** - Video processing3. **`models.py`**: Define the Video model.

- **Uvicorn** - ASGI server   ```python

   from sqlalchemy import Column, Integer, String, Text, TIMESTAMP

## 📦 Cài đặt   from .database import Base



### 1️⃣ **Clone dự án**   class Video(Base):

       __tablename__ = "videos"

```bash

cd E:\PBL6\fastapi-tiktok-clone       id = Column(Integer, primary_key=True, index=True)

```       title = Column(String(255), nullable=False)

       description = Column(Text, nullable=True)

### 2️⃣ **Cài đặt Python dependencies**       url = Column(String(255), nullable=False)

       created_at = Column(TIMESTAMP, server_default='CURRENT_TIMESTAMP')

```bash   ```

pip install -r requirements.txt

```4. **`schemas.py`**: Define the Pydantic schemas.

   ```python

**requirements.txt bao gồm:**   from pydantic import BaseModel

```   from typing import Optional

fastapi==0.104.1

uvicorn[standard]==0.24.0   class VideoBase(BaseModel):

sqlalchemy==2.0.23       title: str

pymysql==1.1.0       description: Optional[str] = None

pydantic==2.5.0       url: str

pydantic-settings==2.1.0

python-jose[cryptography]==3.3.0   class VideoCreate(VideoBase):

passlib[bcrypt]==1.7.4       pass

python-multipart==0.0.6

opencv-python-headless==4.8.1.78   class Video(VideoBase):

email-validator==2.1.0       id: int

```       created_at: str



### 3️⃣ **Cài đặt MySQL**       class Config:

           orm_mode = True

Đảm bảo MySQL đã được cài đặt và đang chạy.   ```



### 4️⃣ **Tạo Database**5. **`main.py`**: Create the FastAPI application.

   ```python

```sql   from fastapi import FastAPI, Depends, HTTPException

CREATE DATABASE pbl6;   from sqlalchemy.orm import Session

```   from . import models, schemas

   from .database import SessionLocal, engine

**Lưu ý:** Các bảng sẽ tự động được tạo khi chạy server lần đầu.

   models.Base.metadata.create_all(bind=engine)

## ⚙ Cấu hình

   app = FastAPI()

### Tạo file `.env`

   def get_db():

Copy từ `.env.example` và cập nhật thông tin:       db = SessionLocal()

       try:

```bash           yield db

# Application       finally:

APP_NAME=TikTok Clone API           db.close()

VERSION=1.0.0

API_V1_STR=/api/v1   @app.post("/videos/", response_model=schemas.Video)

   def create_video(video: schemas.VideoCreate, db: Session = Depends(get_db)):

# Database       db_video = models.Video(**video.dict())

DATABASE_URL=mysql+pymysql://root:1234@localhost:3306/pbl6       db.add(db_video)

       db.commit()

# Security - ĐỔI SECRET_KEY TRONG PRODUCTION!       db.refresh(db_video)

SECRET_KEY=your-secret-key-change-this-in-production-09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7       return db_video

ALGORITHM=HS256

ACCESS_TOKEN_EXPIRE_MINUTES=10080   @app.get("/videos/{video_id}", response_model=schemas.Video)

   def read_video(video_id: int, db: Session = Depends(get_db)):

# File Upload       db_video = db.query(models.Video).filter(models.Video.id == video_id).first()

UPLOAD_DIR=uploads       if db_video is None:

STATIC_DIR=static           raise HTTPException(status_code=404, detail="Video not found")

MAX_UPLOAD_SIZE=104857600       return db_video

```

   @app.get("/videos/", response_model=list[schemas.Video])

**Cập nhật:**   def read_videos(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):

- `DATABASE_URL`: Thay `root:1234` bằng username:password của MySQL bạn       videos = db.query(models.Video).offset(skip).limit(limit).all()

- `SECRET_KEY`: Tạo key mới cho production (dùng `openssl rand -hex 32`)       return videos

   ```

## 🚀 Chạy dự án

### Step 4: Run the Application

### **Khởi động server:**

1. **Run the FastAPI Application**:

```bash   ```bash

python -m uvicorn app.main:app --reload   uvicorn main:app --reload

```   ```



**Giải thích:**2. **Access the API**: Open your browser and go to `http://127.0.0.1:8000/docs` to see the automatically generated API documentation.

- `--reload`: Tự động reload khi code thay đổi (chỉ dùng khi develop)

- Server sẽ chạy tại: **http://127.0.0.1:8000**### Step 5: Testing the API



### **Dừng server:**You can use tools like Postman or the built-in Swagger UI to test the API endpoints:



Nhấn **`Ctrl + C`** trong terminal đang chạy server.- **Create a Video**: POST to `/videos/` with JSON body:

  ```json

### **Production mode (không reload):**  {

      "title": "My First Video",

```bash      "description": "This is a description of my first video.",

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000      "url": "http://example.com/video.mp4"

```  }

  ```

## 📚 API Documentation

- **Get a Video**: GET `/videos/{video_id}`.

Khi server đang chạy, truy cập:

- **Get All Videos**: GET `/videos/`.

- **Swagger UI**: http://127.0.0.1:8000/docs

- **ReDoc**: http://127.0.0.1:8000/redoc### Conclusion

- **OpenAPI JSON**: http://127.0.0.1:8000/openapi.json

You now have a basic FastAPI application connected to a MySQL database for a video-sharing website similar to TikTok. You can expand this project by adding user authentication, video uploads, and more features as needed.
## 🗄 Cấu trúc Database

### **10 Tables chính:**

1. **Users** - Người dùng
   - id, email, username, password, fullName, avatarUrl
   - role (user/admin), status (active/blocked)
   - googleSub (đăng nhập Google)

2. **Videos** - Video
   - id, ownerId, title, description, durationSec
   - url, hlsUrl, thumbUrl
   - visibility (public/hidden/deleted)

3. **Comments** - Bình luận
   - id, userId, videoId, content
   - status (visible/hidden)

4. **Likes** - Thích video
   - userId, videoId (composite key)

5. **Follows** - Theo dõi
   - followerId, followeeId (composite key)

6. **Bookmarks** - Đánh dấu video
   - userId, videoId (composite key)

7. **Messages** - Tin nhắn
   - id, senderId, receiverId, content, mediaUrl
   - status (sent/read/deleted)

8. **Reports** - Báo cáo vi phạm
   - id, reporterId, targetType, targetId, reason
   - status, decision, handledBy

9. **Notifications** - Thông báo
   - id, userId, type, content, refId
   - seen (boolean)

10. **Sessions** - Phiên đăng nhập (nếu cần)

## 🔌 API Endpoints

### **Authentication** (`/api/v1/auth`)
```
POST   /register          - Đăng ký tài khoản
POST   /login             - Đăng nhập (trả về JWT token)
```

### **Users** (`/api/v1/users`)
```
GET    /me                - Lấy thông tin user hiện tại
PUT    /me                - Cập nhật profile
POST   /me/avatar         - Upload avatar
GET    /{user_id}         - Xem profile user khác
GET    /{user_id}/videos  - Lấy videos của user
```

### **Videos** (`/api/v1/videos`)
```
GET    /                  - Danh sách videos (public)
POST   /upload            - Upload video mới
GET    /{video_id}        - Chi tiết video
PUT    /{video_id}        - Cập nhật video (owner only)
DELETE /{video_id}        - Xóa video (owner/admin)
GET    /my-videos         - Videos của tôi
```

### **Comments** (`/api/v1/comments`)
```
GET    /video/{video_id}  - Lấy comments của video
POST   /                  - Tạo comment mới
PUT    /{comment_id}      - Sửa comment
DELETE /{comment_id}      - Xóa comment (soft delete)
```

### **Social** (`/api/v1/social`)
```
POST   /like              - Like video
DELETE /like/{video_id}   - Unlike video
GET    /my-likes          - Videos đã like

POST   /follow            - Follow user
DELETE /follow/{user_id}  - Unfollow user
GET    /followers         - Danh sách followers
GET    /following         - Danh sách following

POST   /bookmark          - Bookmark video
DELETE /bookmark/{video_id} - Bỏ bookmark
GET    /my-bookmarks      - Videos đã bookmark
```

### **Messages** (`/api/v1/messages`)
```
POST   /                  - Gửi tin nhắn
GET    /conversation/{user_id} - Lấy tin nhắn với user
GET    /inbox             - Danh sách cuộc hội thoại
DELETE /{message_id}      - Xóa tin nhắn
```

### **Reports** (`/api/v1/reports`)
```
POST   /                  - Tạo báo cáo
GET    /                  - Danh sách báo cáo (admin)
PUT    /{report_id}/handle - Xử lý báo cáo (admin)
```

### **Notifications** (`/api/v1/notifications`)
```
GET    /                  - Danh sách thông báo
GET    /unseen-count      - Đếm thông báo chưa đọc
PUT    /{notification_id}/seen - Đánh dấu đã đọc
DELETE /{notification_id} - Xóa thông báo
```

## 🧪 Testing

### **Test API với Swagger UI:**

1. Mở http://127.0.0.1:8000/docs
2. Đăng ký user mới: `POST /api/v1/auth/register`
3. Đăng nhập: `POST /api/v1/auth/login` (copy JWT token)
4. Click "Authorize" ở góc phải, paste token: `Bearer <your_token>`
5. Test các API khác

### **Test với curl:**

```bash
# Đăng ký
curl -X POST "http://127.0.0.1:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "password123",
    "fullName": "Test User"
  }'

# Đăng nhập
curl -X POST "http://127.0.0.1:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=password123"

# Lấy profile (với token)
curl -X GET "http://127.0.0.1:8000/api/v1/users/me" \
  -H "Authorization: Bearer <your_token>"
```

### **Run unit tests:**

```bash
pytest tests/
```

## 📁 Cấu trúc dự án

```
fastapi-tiktok-clone/
├── .env                      # Cấu hình môi trường
├── .env.example             # Mẫu cấu hình
├── README.md                # Tài liệu này
├── requirements.txt         # Python dependencies
├── alembic.ini             # Database migration config
│
├── app/
│   ├── __init__.py
│   ├── main.py             # Entry point, khởi tạo FastAPI app
│   ├── database.py         # Database connection & session
│   ├── config.py           # Legacy config (không dùng)
│   │
│   ├── api/
│   │   ├── deps.py         # Dependencies (get_db, get_current_user)
│   │   └── v1/
│   │       ├── auth.py           # Authentication endpoints
│   │       ├── users.py          # User management
│   │       ├── videos.py         # Video CRUD
│   │       ├── comments.py       # Comments
│   │       ├── social.py         # Likes, Follows, Bookmarks
│   │       ├── messages.py       # Direct messages
│   │       ├── reports.py        # Report violations
│   │       └── notifications.py  # Notifications
│   │
│   ├── core/
│   │   ├── config.py       # Settings (Pydantic BaseSettings)
│   │   └── security.py     # JWT, password hashing
│   │
│   ├── models/
│   │   ├── __init__.py     # Import tất cả models
│   │   ├── user.py
│   │   ├── video.py
│   │   ├── comment.py
│   │   ├── like.py
│   │   ├── follow.py
│   │   ├── bookmark.py
│   │   ├── message.py
│   │   ├── report.py
│   │   └── notification.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py         # UserCreate, UserResponse, UserUpdate
│   │   ├── video.py        # VideoCreate, VideoResponse, etc.
│   │   ├── comment.py
│   │   ├── like.py
│   │   ├── follow.py
│   │   ├── bookmark.py
│   │   ├── message.py
│   │   ├── report.py
│   │   └── notification.py
│   │
│   ├── services/
│   │   ├── storage_service.py   # File upload handling
│   │   ├── user_service.py      # User business logic
│   │   └── video_service.py     # Video processing
│   │
│   └── utils/
│       ├── validators.py        # Custom validators
│       └── video_processing.py  # Thumbnail generation
│
├── alembic/
│   ├── env.py              # Alembic environment
│   └── versions/           # Migration files
│
├── static/
│   ├── videos/             # Uploaded videos
│   ├── thumbnails/         # Auto-generated thumbnails
│   └── avatars/            # User avatars
│
├── uploads/
│   └── temp/               # Temporary uploads
│
└── tests/
    ├── conftest.py         # Pytest fixtures
    └── test_api/
        ├── test_auth.py
        ├── test_users.py
        └── test_videos.py
```

## 🔒 Security

- **JWT Authentication**: Tất cả endpoints (trừ register/login) yêu cầu JWT token
- **Password Hashing**: Dùng bcrypt để hash password
- **Role-based Access**: Phân quyền User/Admin
- **File Upload Validation**: Kiểm tra loại file và kích thước
- **SQL Injection Protection**: Dùng SQLAlchemy ORM
- **CORS**: Cấu hình CORS trong production

## 📝 Notes

### **Điều chỉnh trong Production:**

1. **Tắt `--reload`** khi chạy uvicorn
2. **Đổi `SECRET_KEY`** trong `.env`
3. **Cấu hình CORS** đúng domain
4. **Sử dụng HTTPS**
5. **Setup reverse proxy** (Nginx)
6. **Database backup** định kỳ
7. **Monitoring & Logging**

### **Giới hạn hiện tại:**

- Video tối đa **300 giây** (5 phút)
- Upload size tối đa **100MB**
- Token expire sau **7 ngày**

### **Todo / Improvements:**

- [ ] WebSocket cho real-time notifications
- [ ] Redis caching
- [ ] Video transcoding queue (Celery)
- [ ] CDN integration
- [ ] Rate limiting
- [ ] Email verification
- [ ] OAuth2 (Google, Facebook)
- [ ] Video analytics
- [ ] Search & recommendations

## 👨‍💻 Liên hệ

- **Project**: PBL6 - TikTok Clone
- **Tech Stack**: FastAPI + MySQL
- **Database**: pbl6 (10 tables)

---

**Happy Coding! 🚀**
