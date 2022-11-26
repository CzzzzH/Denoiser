import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
from mitsuba.scalar_rgb import ScalarTransform4f as ST

single_plastic = {
    "type": "scene",
    "integrator": {
        "type": "aov",
        "aovs": "depth:depth, normal:sh_normal, position:position, albedo:albedo, sbmc_feature:sbmc_feature",
        "path_tracer": {
            "type": "path",
            "max_depth": 5,
            "rr_depth": 6,
        }
    },
    "PerspectiveCamera": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

single_plastic_s = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 5, "rr_depth": 6, "diffuse_component": False},
    "PerspectiveCamera": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

single_plastic_gt = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 5, "rr_depth": 6, "diffuse_component": True},
    "PerspectiveCamera": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

pinhole = {
    "type": "scene",
    "integrator": {
        "type": "aov",
        "aovs": "depth:depth, normal:sh_normal, position:position, albedo:albedo, sbmc_feature:sbmc_feature",
        "path_tracer": {
            "type": "path",
            "max_depth": 5,
            "rr_depth": 6,
        }
    },
    "PerspectiveCamera": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "mask",
        "opacity": 1.0,
        "material": {
            "type": "dielectric",
            "specular_reflectance": 1.0,
            "specular_transmittance": 1.0,
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    },
    "BSDF1" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "BSDF2" : {
        "type": "twosided",
        "material": {
            "type": "roughconductor",
            "specular_reflectance": 1.0,
            "distribution": "ggx",
            "alpha_u": 0.05,
            "alpha_v": 0.3,
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "Object1": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF1"
        } 
    },
    "Object2": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF2"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

pinhole_s = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 5, "rr_depth": 6, "diffuse_component": False},
    "PerspectiveCamera": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "mask",
        "opacity": 1.0,
        "material": {
            "type": "dielectric",
            "specular_reflectance": 1.0,
            "specular_transmittance": 1.0,
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    },
    "BSDF1" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "BSDF2" : {
        "type": "twosided",
        "material": {
            "type": "roughconductor",
            "specular_reflectance": 1.0,
            "distribution": "ggx",
            "alpha_u": 0.05,
            "alpha_v": 0.3,
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "Object1": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF1"
        } 
    },
    "Object2": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF2"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

pinhole_gt = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 5, "rr_depth": 6, "diffuse_component": True},
    "PerspectiveCamera": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "mask",
        "opacity": 1.0,
        "material": {
            "type": "dielectric",
            "specular_reflectance": 1.0,
            "specular_transmittance": 1.0,
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    },
    "BSDF1" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "BSDF2" : {
        "type": "twosided",
        "material": {
            "type": "roughconductor",
            "specular_reflectance": 1.0,
            "distribution": "ggx",
            "alpha_u": 0.05,
            "alpha_v": 0.3,
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "Object1": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF1"
        } 
    },
    "Object2": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF2"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

depth_of_field = {
    "type": "scene",
    "integrator": {
        "type": "aov",
        "aovs": "depth:depth, normal:sh_normal, position:position, albedo:albedo, sbmc_feature:sbmc_feature",
        "path_tracer": {
            "type": "path",
            "max_depth": 5,
            "rr_depth": 6,
        }
    },
    "PerspectiveCamera": {
        "type": "thinlens",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "aperture_radius": 0.15,
        "focus_distance": 20,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "mask",
        "opacity": 1.0,
        "material": {
            "type": "dielectric",
            "specular_reflectance": 1.0,
            "specular_transmittance": 1.0,
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    },
    "BSDF1" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "BSDF2" : {
        "type": "twosided",
        "material": {
            "type": "roughconductor",
            "specular_reflectance": 1.0,
            "distribution": "ggx",
            "alpha_u": 0.05,
            "alpha_v": 0.3,
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "Object1": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF1"
        } 
    },
    "Object2": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF2"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

depth_of_field_s = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 5, "rr_depth": 6, "diffuse_component": False},
    "PerspectiveCamera": {
        "type": "thinlens",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "aperture_radius": 0.15,
        "focus_distance": 20,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "mask",
        "opacity": 1.0,
        "material": {
            "type": "dielectric",
            "specular_reflectance": 1.0,
            "specular_transmittance": 1.0,
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    },
    "BSDF1" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "BSDF2" : {
        "type": "twosided",
        "material": {
            "type": "roughconductor",
            "specular_reflectance": 1.0,
            "distribution": "ggx",
            "alpha_u": 0.05,
            "alpha_v": 0.3,
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "Object1": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF1"
        } 
    },
    "Object2": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF2"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}

depth_of_field_gt = {
    "type": "scene",
    "integrator": {"type": "path", "max_depth": 5, "rr_depth": 6, "diffuse_component": True},
    "PerspectiveCamera": {
        "type": "thinlens",
        "near_clip": 1.0,
        "far_clip": 1000.0,
        "fov": 45,
        "aperture_radius": 0.15,
        "focus_distance": 20,
        "to_world": T([[-0.00550949, -0.342144, -0.939631, 23.895],
                     [1.07844e-005, 0.939646, -0.342149, 11.2207],
                     [0.999985, -0.00189103, -0.00519335, 0.0400773],
                     [0, 0, 0, 1]]),
        "sampler": {
            "type": "independent",
            "sample_count": 4
        },            
        "film": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "tent"
            },
            "file_format": "openexr",
            "pixel_format": "rgb",
            "width": 128,
            "height": 128
        }
    },
    "BSDF0" : {
        "type": "mask",
        "opacity": 1.0,
        "material": {
            "type": "dielectric",
            "specular_reflectance": 1.0,
            "specular_transmittance": 1.0,
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    },
    "BSDF1" : {
        "type": "twosided",
        "material": {
            "type": "plastic",
            "diffuse_reflectance": {
                "type": "rgb",
                "value": [0.9, 0.9, 0.9]
            },
            "int_ior": 1.5,
            "ext_ior": 1.0,
            "nonlinear": True
        }
    },
    "BSDF2" : {
        "type": "twosided",
        "material": {
            "type": "roughconductor",
            "specular_reflectance": 1.0,
            "distribution": "ggx",
            "alpha_u": 0.05,
            "alpha_v": 0.3,
        }
    },
    "FloorBSDF" : {
        "type": "twosided",
        "material": {
            "type": "diffuse",
            "reflectance": {
                "type": "checkerboard",
                "color0": {
                    "type": "rgb",
                    "value": [0.325, 0.31, 0.25]
                },
                "color1":{
                    "type": "rgb",
                    "value": [0.725, 0.71, 0.68]
                },
                "to_uv": ST.scale([10.0, 10.0, 0.0])
            }
        }
    },
    "Floor": {
        "type": "rectangle",
        "to_world": T([[-39.9766, 39.9766, -1.74743e-006, 0], 
                     [4.94249e-006, 2.47125e-006, -56.5355, 0],
                     [-39.9766, -39.9766, -5.2423e-006, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "FloorBSDF"
        } 
    },
    "Object0": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF0"
        } 
    },
    "Object1": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF1"
        } 
    },
    "Object2": {
        "type": "obj",
        "filename": "assets/teapot/models/Mesh001.obj",
        "to_world": T([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
        "bsdf": {
            "type" : "ref",
            "id" : "BSDF2"
        } 
    },
    "EnvironmentMapEmitter": {
        "type": "envmap",
        "to_world": T([[-0.922278, 0, 0.386527, 0],
                     [0, 1, 0, 0],
                     [-0.386527, 0, -0.922278, 1.17369],
                     [0, 0, 0, 1]]),
        "filename": "D:/envmap.hdr"
    }
}