from pydantic import BaseModel
from typing import Optional

class PacienteBase(BaseModel):
    EDAD: float
    IMC: float
    SEXO: float
    COLESTEROL: float
    ESTRATO: float
    NIVEL_EDUCATIVO: float
    ACCESO_ELECTRICO: float
    ACUEDUCTO: float
    ALCANTARILLADO: float
    GAS_NATURAL: float
    ANTECEDENTES_FAMILIARES: float
    FUMADOR: float
    DEPARTAMENTO: str
    MUNICIPIO: str
    ESTADO_CIVIL: str
    AREA: str
    INTERNET: str
    ETNIA: str
    OCUPACION: str

class PrediccionResponse(BaseModel):
    riesgo_cardiovascular: int
    puntaje_riesgo: float
    probabilidades: list[float]