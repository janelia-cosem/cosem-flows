from enum import Enum
from cosem_flows.attrs import ContentTypeEnum
from pydantic import BaseModel
from typing import Dict, Sequence
import click

class ClassNameEnum(str, Enum):
    fibsem_uint8 = "fibsem-uint8"
    fibsem_uint16 = "fibsem-uint16"
    gt = "gt"
    cent = "cent"
    cent_dapp = "cent-dapp"
    cent_sdapp = "cent-sdapp"
    chrom = "chrom"
    er = "er"
    er_palm = "er_palm"
    er_sim = "er_sim"
    er_mem = "er-mem"
    eres = "eres"
    eres_mem = "eres-mem"
    endo = "endo"
    endo_mem = "endo-mem"
    echrom = "echrom"
    ecs = "ecs"
    golgi = "golgi"
    golgi_mem = "golgi-mem"
    hchrom = "hchrom"
    ld = "ld"
    ld_mem = "ld-mem"
    lyso = "lyso"
    lyso_mem = "lyso-mem"
    mt = "mt"
    mt_in = "mt-in"
    mt_out = "mt-out"
    mito = "mito"
    mito_palm = "mito_palm"
    mito_sim = "mito_sim"
    mito_mem = "mito-mem"
    mito_ribo = "mito-ribo"
    ne = "ne"
    ne_mem = "ne-mem"
    np = "np"
    np_in = "np-in"
    np_out = "np-out"
    nucleolus = "nucleolus"
    nechrom = "nechrom"
    nhchrom = "nhchrom"
    nucleus = "nucleus"
    pm = "pm"
    ribo = "ribo"
    vesicle = "vesicle"
    vesicle_mem = "vesicle-mem"
    er_ribo_contacts = "er_ribo_contacts"
    er_golgi_contacts = "er_golgi_contacts"
    er_mito_contacts = "er_mito_contacts"
    endo_er_contacts = "endo_er_contacts"
    er_nucleus_contacts = "er_nucleus_contacts"
    er_pm_contacts = "er_pm_contacts"
    er_vesicle_contacts = "er_vesicle_contacts"
    golgi_vesicle_contacts = "golgi_vesicle_contacts"
    endo_golgi_contacts = "endo_golgi_contacts"
    mito_pm_contacts = "mito_pm_contacts"
    er_mt_contacts = "er_mt_contacts"
    endo_mt_contacts = "endo_mt_contacts"
    golgi_mt_contacts = "golgi_mt_contacts"
    mito_mt_contacts = "mito_mt_contacts"
    mt_nucleus_contacts = "mt_nucleus_contacts"
    mt_vesicle_contacts = "mt_vesicle_contacts"
    mt_pm_contacts = "mt_pm_contacts"
    mito_skeleton = "mito_skeleton"
    mito_skeleton_lsp = "mito_skeleton-lsp"
    er_medial_surface = "er_medial-surface"
    er_curvature = "er_curvature"
    ribo_classified = "ribo_classified"


class ClassMetadata(BaseModel):
    description: str
    contentTypes: Sequence[ContentTypeEnum]
    color: str


class ClassMetadataCollection(BaseModel):
    classes: Dict[ClassNameEnum, ClassMetadata]

class_info = ClassMetadataCollection(
    classes={
        "cent": ClassMetadata(
            description="Centrosome",
            contentTypes=["segmentation", "prediction"],
            color="white",
        ),
        "cent-dapp": ClassMetadata(
            description="Centrosome Distal Appendage",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "cent-sdapp": ClassMetadata(
            description="Centrosome Subdistal Appendage",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "chrom": ClassMetadata(
            description="Chromatin",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "er": ClassMetadata(
            description="Endoplasmic Reticulum",
            color="blue",
            contentTypes=["segmentation", "prediction"],
        ),
        "er_palm": ClassMetadata(
            description="Light Microscopy (PALM) of ER",
            color="white",
            contentTypes=["lm"],
        ),
        "er_sim": ClassMetadata(
            description="Light Microscopy (SIM) of the ER",
            color="white",
            contentTypes=["lm"],
        ),
        "er-mem": ClassMetadata(
            description="Endoplasmic Reticulum membrane",
            color="blue",
            contentTypes=["segmentation", "prediction"],
        ),
        "eres": ClassMetadata(
            description="Endoplasmic Reticulum Exit Site",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "eres-mem": ClassMetadata(
            description="Endoplasmic Reticulum Exit Site membrane",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "endo": ClassMetadata(
            description="Endosomal Network",
            color="magenta",
            contentTypes=["segmentation", "prediction"],
        ),
        "endo-mem": ClassMetadata(
            description="Endosome membrane",
            color="magenta",
            contentTypes=["segmentation", "prediction"],
        ),
        "echrom": ClassMetadata(
            description="Euchromatin",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "ecs": ClassMetadata(
            description="Extracellular Space",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "golgi": ClassMetadata(
            description="Golgi",
            color="cyan",
            contentTypes=["segmentation", "prediction"],
        ),
        "golgi-mem": ClassMetadata(
            description="Golgi membrane",
            color="cyan",
            contentTypes=["segmentation", "prediction"],
        ),
        "hchrom": ClassMetadata(
            description="Heterochromatin",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "ld": ClassMetadata(
            description="Lipid Droplet",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "ld-mem": ClassMetadata(
            description="Lipid Droplet membrane",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "lyso": ClassMetadata(
            description="Lysosome",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "lyso-mem": ClassMetadata(
            description="Lysosome membrane",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "mt": ClassMetadata(
            description="Microtubule",
            color="orange",
            contentTypes=["segmentation", "prediction"],
        ),
        "mt-in": ClassMetadata(
            description="Microtubule inner",
            color="orange",
            contentTypes=["segmentation", "prediction"],
        ),
        "mt-out": ClassMetadata(
            description="Microtubule outer",
            color="orange",
            contentTypes=["segmentation", "prediction"],
        ),
        "mito": ClassMetadata(
            description="Mitochondria",
            color="green",
            contentTypes=["segmentation", "prediction"],
        ),
        "mito_palm": ClassMetadata(
            description="Light Microscopy (PALM) of Mitochondria",
            color="white",
            contentTypes=["lm"],
        ),
        "mito_sim": ClassMetadata(
            description="Light Microscopy (SIM) of Mitochondria",
            color="white",
            contentTypes=["lm"],
        ),
        "mito-mem": ClassMetadata(
            description="Mitochondria membrane",
            color="green",
            contentTypes=["segmentation", "prediction"],
        ),
        "mito-ribo": ClassMetadata(
            description="Mitochondria Ribosome",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "ne": ClassMetadata(
            description="Nuclear Envelope",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "ne-mem": ClassMetadata(
            description="Nuclear Envelope membrane",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "np": ClassMetadata(
            description="Nuclear Pore",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "np-in": ClassMetadata(
            description="Nuclear Pore inner",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "np-out": ClassMetadata(
            description="Nuclear Pore outer",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "nucleolus": ClassMetadata(
            description="Nucleoulus",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "nechrom": ClassMetadata(
            description="Nucleoulus associated Euchromatin",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "nhchrom": ClassMetadata(
            description="Nucleoulus associated Heterochromatin",
            color="white",
            contentTypes=["segmentation", "prediction"],
        ),
        "nucleus": ClassMetadata(
            description="Nucleus",
            color="red",
            contentTypes=["segmentation", "prediction"],
        ),
        "pm": ClassMetadata(
            description="Plasma Membrane",
            color="orange",
            contentTypes=["segmentation", "prediction"],
        ),
        "ribo": ClassMetadata(
            description="Ribosome",
            color="yellow",
            contentTypes=["segmentation", "prediction"],
        ),
        "vesicle": ClassMetadata(
            description="Vesicle",
            color="red",
            contentTypes=["segmentation", "prediction"],
        ),
        "vesicle-mem": ClassMetadata(
            description="Vesicle membrane",
            color="red",
            contentTypes=["segmentation", "prediction"],
        ),
        "er_ribo_contacts": ClassMetadata(
            description="ER - Ribosome Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_golgi_contacts": ClassMetadata(
            description="ER - Golgi Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_mito_contacts": ClassMetadata(
            description="ER - Mito Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "endo_er_contacts": ClassMetadata(
            description="Endosome - ER Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_nucleus_contacts": ClassMetadata(
            description="ER - Nucleus Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_pm_contacts": ClassMetadata(
            description="ER - Plasma Membrane Contat Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_vesicle_contacts": ClassMetadata(
            description="ER - Vesicle Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "golgi_vesicle_contacts": ClassMetadata(
            description="Golgi - Vesicle Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "endo_golgi_contacts": ClassMetadata(
            description="Endosome - Golgi Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "mito_pm_contacts": ClassMetadata(
            description="Mito - Plasma Membrane Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_mt_contacts": ClassMetadata(
            description="ER - Microtubule Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "endo_mt_contacts": ClassMetadata(
            description="Endosome - Microtubule Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "golgi_mt_contacts": ClassMetadata(
            description="Golgi - Microtubule Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "mito_mt_contacts": ClassMetadata(
            description="Mitochondria - Microtubule Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "mt_nucleus_contacts": ClassMetadata(
            description="Microtubule - Nucleus Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "mt_vesicle_contacts": ClassMetadata(
            description="Microtubule - Vesicle Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "mt_pm_contacts": ClassMetadata(
            description="Microtubule - Plasma Membrane Contact Sites",
            color="white",
            contentTypes=["analysis"],
        ),
        "mito_skeleton": ClassMetadata(
            description="Mitochrondria Skeletons",
            color="white",
            contentTypes=["analysis"],
        ),
        "mito_skeleton-lsp": ClassMetadata(
            description="Mitochrondria Skeletons: Longest Shortest Path",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_medial-surface": ClassMetadata(
            description="ER Medial Surface",
            color="white",
            contentTypes=["analysis"],
        ),
        "er_curvature": ClassMetadata(
            description="Reconstructed ER from Medial Surface with Curvature",
            color="white",
            contentTypes=["analysis"],
        ),
        "ribo_classified": ClassMetadata(
            description="Ribosomes classified by contact surface",
            color="white",
            contentTypes=["analysis"],
        ),
        "fibsem-uint8": ClassMetadata(
            description="FIB-SEM Data (compressed)", color="white", contentTypes=["em"]
        ),
        "fibsem-uint16": ClassMetadata(
            description="FIB-SEM Data (uncompressed)",
            color="white",
            contentTypes=["em"],
        ),
        "gt": ClassMetadata(
            description="Ground truth", color="white", contentTypes=["segmentation"]
        ),
    }
)

@click.command()
@click.option("-s", "--schema", required=False, type=bool, is_flag=True)
def main(schema: bool) -> None:
    if schema:
        print(class_info.schema_json(indent=2))
    else:
        print(class_info.json(indent=2))


if __name__ == '__main__':
    main()