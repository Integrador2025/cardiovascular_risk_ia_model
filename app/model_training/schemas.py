from pydantic import BaseModel
from typing import Optional

class PacienteInput(BaseModel):
    DEPARTAMENTO: str
    MUNICIPIO: str
    SEXO: int
    EDAD: float
    ESTADO_CIVIL: str
    NIVEL_EDUCATIVO: str
    ESTRATO: str
    AREA: str
    ACCESO_ELECTRICO: int
    ACUEDUCTO: int
    ALCANTARILLADO: int
    GAS_NATURAL: int
    INTERNET: int
    ETNIA: str
    OCUPACION: str
    IMC: float
    ANTECEDENTES_FAMILIARES: int
    FUMADOR: int
    COLESTEROL: float
