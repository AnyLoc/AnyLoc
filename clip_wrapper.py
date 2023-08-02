# Contains a wrapper around CLIP backends
"""
    Currently the following backends are supported (and tested)
    1. Official OpenAI CLIP [1]: "openai": IMPL_OPENAI
    2. Open Source CLIP [2]: "open_clip": IMPL_OPEN_CLIP
    
    
    [1]: https://github.com/openai/CLIP
    [2]: https://github.com/mlfoundations/open_clip
"""

# %%
# Python path gimmick
import os
import sys
from pathlib import Path
# Set the "./" from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARNING: __file__ not found, trying local")
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f"{Path(dir_name)}")
# Add to path
print("[INFO]: CLIP Wrapper is modifying path")
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")


# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image
from typing import Union
# CLIP implementations
import clip
import open_clip
# Cache
from configs import caching_directory
from utilities import to_pil_list


# %%
# Main clip wrapper class
class ClipWrapper:
    r"""
        A wrapper around the following CLIP implementations
        - IMPL_OPENAI: Official OpenAI implementation [1]
        - IMPL_OPEN_CLIP: Open-source implementation [2]
        
        [1]: https://github.com/openai/CLIP
        [2]: https://github.com/mlfoundations/open_clip
        
        Usage:
        ```py
        # ----------------- Testing OpenAI CLIP -----------------
        model = ClipWrapper(ClipWrapper.IMPL_OPENAI, "ViT-B/32", 
                    device="cuda")
        img = Image.open("./dir-ignore/CLIP.png")
        text = ["a diagram", "a dog", "a cat"]
        with torch.no_grad():
            img_features = model.encode_image(img)
            text_features = model.encode_text(text)
            probs, img_f, txt_f = model(img, text)
        # If you want to modify other things
        tokenizer = model.get_tokenizer(False)
        preprocessing = model.get_preprocessing(False)
        # ----------------- Testing Open CLIP -----------------
        model = ClipWrapper(ClipWrapper.IMPL_OPEN_CLIP, 
                    "ViT-B-32-quickgelu", pretrained='laion400m_e32', 
                    device="cuda")
        # ----- List all model backbones and pretrained datasets -----
        model.list_models()
        ```
        
        Caching policy: `use_caching != False` (True or `exp_id`)
        - `.models/clip`: For storing checkpoints of CLIP models
        - `experiments/exp_id`: For specific experiment ID
        - `images` for images and `text` for text. Stored globally if
            `exp_id` is None (when `use_caching = True`). All cached
            Tensors have `requires_grad=False`.
        - If `base_cache_dir` is None then models are stored in 
            `~/.cache/clip` (no images or text). Otherwise, all cache
            is stored in the base cache directory.
        
        Constructor parameters:
        - impl:     Implementation (in the implementations list above)
        - name:     Model name, must be in `list_models`
        - pretrained:   Pretraining dataset (only for IMPL_OPEN_CLIP).
                        In that implementation, None means default.
                        Not used for IMPL_OPENAI
        - prep_apply:   Apply preprocessing pipeline to input images 
                        and text. Setting this to False means that you 
                        have to explicitly call the 'preprocessing' 
                        and 'tokenizing' pipelines before calling the 
                        encoding functions.
        - use_caching:      If True, cache directory in config is used
                            to store the models and the result cache.
                            If you pass a string (or equivalent), it 
                            treats this as a parent directory for 
                            results cache (not model), treat this as 
                            an `experiment identifier` (it can contain
                            '/'). This turns ON caching.
        - base_cache_dir:   Folder for default cache (for model). If
                            caching is ON, this is also the base 
                            directory for the images and text cache.
        - save_norm_descs:  If True, the cached descriptors are 
                            normalized before writing to cache. This
                            also means that normalized descriptors are
                            read (from the same cache). Only if 
                            parameter  `use_caching` is configured for 
                            saving cache.
        - **clip_kwargs:    Keyword arguments for the CLIP constructor
                            used in the backend. The class 'device' 
                            from here.
                            - cache_dir: It is set appropriately
    """
    # Variables
    IMPL_OPENAI = "openai"
    IMPL_OPEN_CLIP = "open_clip"
    # Functions
    def __init__(self, impl, name, pretrained=None, prep_apply=True,
            use_caching:Union[str,int,bool]=True, 
            base_cache_dir:Union[Path,None]=caching_directory, 
            save_norm_descs=True, **clip_kwargs) -> None:
        # Caching variables
        self._use_caching = False
        self._md = self._rdi = self._rdt = None # Model, image, text
        self._ensure_cache_dir(use_caching, base_cache_dir)
        self.norm_cache = save_norm_descs
        # Implementation
        assert impl in [self.IMPL_OPEN_CLIP, self.IMPL_OPENAI]
        self.impl = impl
        print(f"Using CLIP implementation from: {self.impl}")
        self.clip: nn.Module = None # CLIP model
        self.preprocessing: tvf.Compose = None  # Image preprocessing
        self.tokenizer = None   # Text tokenizer
        if self.impl == self.IMPL_OPENAI:
            self.clip, self.preprocessing = clip.load(name, 
                    download_root=self._md, **clip_kwargs)
            self.tokenizer = lambda x, clen=77: clip.tokenize(x, clen)
        elif self.impl == self.IMPL_OPEN_CLIP:
            # assert pretrained is not None
            self.clip, _, self.preprocessing = open_clip.\
                create_model_and_transforms(name, 
                        pretrained=pretrained, cache_dir=self._md, 
                        **clip_kwargs)
            self.tokenizer = open_clip.get_tokenizer(name)
        # Preprocessing instruction
        self.prep_img = prep_apply
        self.prep_text = prep_apply
        # Other things
        self.device = torch.device(clip_kwargs.get("device", "cpu"))

    def _ensure_cache_dir(self, use_caching=True, 
                base_dir:str=caching_directory):
        """
            Sort out everything related to cache and set up dirs.
            Sets the correct values for
            - self._use_caching: True | False
            - self._md: Model directory
            - self._rdi: Images directory
            - self._rdt: Text directory
            
            Parameters:
            - use_caching: True | False | str-like (experiment id)
            - base_dir: str (path) | None (no image & text)
        """
        _ex = lambda x: os.path.realpath(os.path.expanduser(x))
        # Models directory (for storing models)
        _md = _ri = _rt = None
        _base_dir = _ex("~/.cache/clip")
        if base_dir is not None:
            _base_dir = _ex(str(base_dir))
            _md = f"{_base_dir}/.models/clip"
        else:
            _md = _base_dir
        if not os.path.isdir(_md):
            os.makedirs(_md)
        print(f"Model cache: {_md}")    # Have to use model cache
        if type(use_caching) != bool:   # Experiment identifier
            _base_dir = _ex(f"{_base_dir}/experiments/{use_caching}")
        if use_caching:
            _ri, _rt = f"{_base_dir}/images", f"{_base_dir}/text"
            if not os.path.isdir(_ri):
                os.makedirs(_ri)
            if not os.path.isdir(_rt):
                os.makedirs(_rt)
            self._use_caching = True
            print(f"Image cache directory: {_ri}")
            print(f"Text cache directory: {_rt}")
        else:
            print("Image and text cache will not be used")
            self._use_caching = False
        self._md, self._rdi, self._rdt = _md, _ri, _rt

    def get_tokenizer(self, disable_prep=True):
        """
            Returns the tokenizer object for the text tokenization.
            
            Parameters:
            - disable_prep: If True, the preprocessing for text is
                            disabled (hereon) by setting the
                            `prep_text=False`. If False, the setting
                            is not changed.
            
            Returns:
            - tokenizer:    Default tokenizer that's used to tokenize
                            list of strings.
        """
        self.prep_text = False if disable_prep else self.prep_text
        return self.tokenizer

    def get_preprocessing(self, disable_prep=True):
        """
            Returns the image preprocessing (torchvision Compose)
            pipeline.
            
            Parameters:
            - disable_prep: If True, the preprocessing for image is
                            disabled (hereon) by setting the 
                            `prep_img=False`. If False, the setting
                            is not changed.
            
            Returns:
            - preprocessing:    Preprocessing function for images
        """
        self.prep_img = False if disable_prep else self.prep_img
        return self.preprocessing

    def encode_image(self, image, normalize=False, ci=None):
        """
            Calls `encode_image` of the CLIP implementation. 
            - If `load` was used (that is, `prep_img = False`), then 
                preprocessing has to be applied before using this.
            - If `prep_img = True`, it is moved to the appropriate
                device before passing it to the implementation.
            
            Parameters:
            - image:    An image (after preprocessing if the setting 
                        `prep_img=False`). Preprocessing is applied
                        only if `prep_img=True`.
                        - If type `PIL.Image.Image` then default 
                            preprocessing is applied.
                        - For any other type, a list of PIL.Image 
                            objects is constructed and preprocessing 
                            is applied.
            - normalize:    If True, normalize the image features
            - ci:   A str-like identifier for caching. If None, no
                    caching is used (should be unique for each image).
            
            Returns:
            - img_features: Image descriptors of shape [N=1, D=512]
            
            Note:
            - If `ci` is found, the forward pass is not used since the 
                cache is directly returned. `image` can be None if 
                you're _sure_ the cache exists.
            - When writing to cache, `normalize` plays a role.
        """
        # See if cache is present
        if self._use_caching and (ci is not None):
            _cir = f"{self._rdi}/{ci}.pt"
            if os.path.isfile(_cir):
                img_features = torch.load(_cir, self.device)
                # Cast Torch Tensor to float type, TODO This can be removed later
                img_features = img_features.type(torch.float32)
                if normalize:
                    img_features = F.normalize(img_features, dim=-1)
                return img_features
            elif image is None:
                raise FileNotFoundError(f"{_cir}")
        # Preprocess image
        if self.prep_img:
            if type(image) == Image.Image:
                image = self.preprocessing(image).unsqueeze(0)\
                        .to(self.device)
            else:   # Create batch of PIL.Image objects
                pil_images = to_pil_list(image)
                imgs_ret = []
                for pil_img in pil_images:
                    imgs_ret.append(self.preprocessing(pil_img)\
                            .unsqueeze(0).to(self.device))
                # Now change to [B, C, H, W] tensor
                image = torch.concatenate(imgs_ret, dim=0)
        # CLIP image features, image is [B, C, H, W]
        img_features: torch.Tensor = self.clip.encode_image(image)
        # Cast Torch Tensor to float type
        img_features = img_features.type(torch.float32)
        # Normalize CLIP image features
        if normalize:
            img_features = F.normalize(img_features, dim=-1)
        # Write results to cache
        if self._use_caching and (ci is not None):
            _cis = f"{self._rdi}/{ci}.pt"
            if self.norm_cache:
                img_features = F.normalize(img_features, dim=-1)
            torch.save(img_features.detach().cpu(), _cis)
        return img_features

    def encode_text(self, text, context_length=77, normalize=False, 
            ci=None):
        """
            Calls `encode_text` of the CLIP implementation.
            - If `load` was used (that is, `prep_text = False`), then
                tokenization has to be applied before using this.
            - If `prep_text = True`, then tokenization is applied to
                the passed text (which is assumed as list of labels),
                and the tensor is model to the appropriate device 
                before passing it to the implementation.
            
            Parameters:
            - text:     If `prep_text = False` then should be the
                        tokenized text tokens (tensor), else if the
                        `prep_text = True`, then should be a list of
                        str (containing "text phrases")
            - context_length:   Context length for token dimensions
            - normalize:    Normalize the CLIP encodings
            - ci:   A str-like identifier for caching. If None, no
                    caching is used (should be unique for each text).
            
            Returns:
            - text_features:    Text embeddings of shape [N, D=512]
            
            Note:
            - If `ci` is found, no `context_length` and forward pass 
                is used since the cache is directly returned. `text` 
                can be None if you're _sure_ the cache exists.
            - When writing to cache, `normalize` plays a role.
            - `context_length` is used only if `self.prep_text=True`
        """
        # See if cache is present
        if self._use_caching and (ci is not None):
            _ctr = f"{self._rdt}/{ci}.pt"
            if os.path.isfile(_ctr):
                text_features = torch.load(_ctr, self.device)
                if normalize:
                    text_features = F.normalize(text_features, dim=-1)
                return text_features
            elif text is None:
                raise FileNotFoundError(f"{_ctr}")
        # Preprocess text
        if self.prep_text:
            text = self.tokenizer(text, context_length)\
                    .to(self.device)
        # CLIP text features
        text_features: torch.Tensor = self.clip.encode_text(text)
        # Normalize CLIP text features
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        # Write results to cache
        if self._use_caching and (ci is not None):
            _ctr = f"{self._rdt}/{ci}.pt"
            if self.norm_cache:
                text_features = F.normalize(text_features, dim=-1)
            torch.save(text_features.detach().cpu(), _ctr)
        return text_features

    def __call__(self, img, text, normalize=False, context_length=77,
                ci=None, detach=True):
        """
            Calls the implementation model to return the class scores
            (probability that `img` belongs to an item in `text`) 
            along with the descriptors for image and text.
            
            Parameters:
            - img:  The image input for the `encode_image` function.
                    Note the `prep_img` setting.
            - text: The text input for the `encode_text` function.
                    Note the `prep_text` setting.
            - normalize:    Normalize the CLIP encodings (image and 
                            text both)
            - context_length:   Context length for token dimensions
            - ci:   Cache identifier (should be unique per query). If
                    None, then caching is not used.
            - detach:   If True, then returned variables are detached.
            
            Returns:
            - probs:    A [N_img, N_text] probability distribution
            - img_features: Image CLIP features
            - text_features:    CLIP embeddings for text
        """
        # Forward pass
        img_features = self.encode_image(img, normalize, ci=ci)
        text_features = self.encode_text(text, context_length, 
                            normalize, ci=ci)
        # Get logits for classes [N_imgs, N_text]: text class matrix
        logits_per_img = 100 * (text_features @ img_features.T).T
        probs = logits_per_img.softmax(dim=-1)
        if detach:
            probs, img_features, text_features = map(lambda x: \
                    x.detach(), [probs, img_features, text_features])
        return probs, img_features, text_features

    @staticmethod
    def load(impl, name, pretrained=None, **kwargs):
        """
            Loads a CLIP method using the available implementations
            - ClipWrapper.IMPL_OPENAI
            - ClipWrapper.IMPL_OPEN_CLIP
            See the class doc for more.
            
            Parameters:
            - impl: Implementation (in the list above)
            - name: Model name, must be in `list_models`
            - pretrained:   Dataset (for IMPL_OPEN_CLIP)
                            None (ignored) for IMPL_OPENAI
            - **kwargs:     Keyword arguments for constructor
            
            Returns Tuple:
            - wrapper:          The CLIP wrapper with prep_apply=False
            - preprocessing:    The image pre-processing function
            
            See class constructor for more information
        """
        wrapper = ClipWrapper(impl, name, pretrained, 
                    prep_apply=False, **kwargs)
        return wrapper, wrapper.preprocessing

    @staticmethod
    def list_models(ret_vals=False):
        """
            Lists all model names for the implementations
            
            Parameters:
            - ret_vals: If True, return a dictionary containing models
            
            Returns (if `ret_vals`):
            - models_dict:  Keys are IMPL_*, values are list
        """
        # OpenAI
        models = clip.available_models()
        print("- IMPL_OPENAI - name")
        for model in models:
            print(f"    - \"{model}\"")
        # Open CLIP
        descs = open_clip.list_pretrained()
        print(f"- IMPL_OPEN_CLIP - name | pretrained")
        for model, pret in descs:
            print(f"    - \"{model}\" | \"{pret}\"")
        if ret_vals:
            return {
                ClipWrapper.IMPL_OPENAI: models,
                ClipWrapper.IMPL_OPEN_CLIP: descs
            }


# %%
# Entry point for python (not jupyter / ipython)
if __name__ == "__main__" and (not "ipykernel" in sys.argv[0]):
    print(f"WARNING: Do NOT run this wrapper as main")
    quit(0)


# %%
# Experimental section

# %%
