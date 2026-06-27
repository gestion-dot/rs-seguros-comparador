import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET", "rs-seguros-secret-2024")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

security = HTTPBearer()


def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Token inválido")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")


def check_credentials(username: str, password: str) -> bool:
    expected_user = os.getenv("APP_USERNAME", "rsseguros")
    expected_pass = os.getenv("APP_PASSWORD", "FINANZAS")
    return username == expected_user and password == expected_pass
