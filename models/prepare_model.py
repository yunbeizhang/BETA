import torchvision.models as models
import torch
import timm
import re


import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
from typing import List, Union
import os

# Import the class names from the provided file
# Note: You'll need to have the cls_names.py file in the same directory or adjust the import path
imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]


def get_class_names(dataset_name: str):
    # get the class names
    if "imagenet" in dataset_name:
        if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
            label_mask = eval(f"{dataset_name.upper()}_MASK")
            if dataset_name == "imagenet_r":
                class_names = [imagenet_classes[i] for i, m in enumerate(label_mask) if m]
            else:
                class_names = [imagenet_classes[i] for i in label_mask]
        else:
            class_names = imagenet_classes
    elif "cifar" in dataset_name:
        class_names = eval(f"{dataset_name.split('_')[0]}_classes")
    else:
        class_names = eval(f"{dataset_name}_classes")

    # post-process the class names
    class_names = [name.replace("_", " ") for name in class_names]
    return class_names


IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

class CLIPZeroShotClassifier(nn.Module):
    """
    A zero-shot classifier using OpenAI's CLIP model.
    Converts class names to text embeddings using multiple templates and averaging them.
    """
    
    def __init__(self, dataset_name: str = "imagenet", backbone: str = "ViT-B/16", device: str = None, use_templates: bool = True):
        super().__init__()
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Validate backbone
        available_models = clip.available_models()
        if backbone not in available_models:
            raise ValueError(f"Backbone '{backbone}' not available. Available models: {available_models}")
            
        # Load CLIP model
        self.backbone = backbone
        self.model, self.preprocess = clip.load(backbone, device=self.device)
        
        # Get class names for the specified dataset
        self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)
        self.use_templates = use_templates
        
        # Create text embeddings for all classes
        self.text_embeddings = self._create_text_embeddings()
        
        print(f"Initialized CLIP zero-shot classifier for {dataset_name}")
        print(f"Backbone: {self.backbone}")
        print(f"Using template ensembling: {self.use_templates}")
        if self.use_templates:
            print(f"Number of templates: {len(IMAGENET_TEMPLATES)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Device: {self.device}")
    
    def _create_text_embeddings(self) -> torch.Tensor:
        """
        Create text embeddings for all class names using template ensembling or single template.
        
        Returns:
            torch.Tensor: Normalized text embeddings of shape (num_classes, embedding_dim)
        """
        if self.use_templates:
            # Use template ensembling with multiple prompts
            all_embeddings = []
            
            print("Creating text embeddings with template ensembling...")
            for class_name in self.class_names:
                # Create prompts for this class using all templates
                class_prompts = [template.format(class_name) for template in IMAGENET_TEMPLATES]
                
                # Tokenize all prompts for this class
                class_tokens = clip.tokenize(class_prompts).to(self.device)
                
                # Generate embeddings for all templates of this class
                with torch.no_grad():
                    class_embeddings = self.model.encode_text(class_tokens)
                    # Normalize each embedding
                    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                    # Average across all templates for this class
                    averaged_embedding = class_embeddings.mean(dim=0)
                    # Normalize the averaged embedding
                    averaged_embedding = averaged_embedding / averaged_embedding.norm()
                
                all_embeddings.append(averaged_embedding)
            
            # Stack all class embeddings
            text_embeddings = torch.stack(all_embeddings)
            
        else:
            # Use single template: "a photo of a {class_name}."
            text_prompts = [f"a photo of a {class_name}." for class_name in self.class_names]
            
            # Tokenize the text prompts
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            
            # Generate text embeddings
            with torch.no_grad():
                text_embeddings = self.model.encode_text(text_tokens)
                # Normalize the embeddings
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for CLIP model.
        
        Args:
            image: Can be a file path (str), PIL Image, or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, str):
            # Load image from file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path, PIL Image, or numpy array")
        
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            images: Batch of preprocessed images
            
        Returns:
            torch.Tensor: Logits for each class
        """
        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = self.model.encode_image(images)
            # Normalize the embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (logits) between image and text embeddings
        # This is equivalent to cosine similarity scaled by the model's temperature
        logits = (image_embeddings @ self.text_embeddings.T) * self.model.logit_scale.exp()
        
        return logits
    
    def predict(self, image: Union[str, Image.Image, np.ndarray], top_k: int = 5) -> List[tuple]:
        """
        Predict the class of an image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples: (class_name, probability) for top_k predictions
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image)
        
        # Get logits
        logits = self.forward(image_tensor)
        
        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        # Convert to list of (class_name, probability) tuples
        predictions = []
        for i in range(top_k):
            class_idx = top_indices[i].item()
            prob = top_probs[i].item()
            class_name = self.class_names[class_idx]
            predictions.append((class_name, prob))
        
        return predictions
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> torch.Tensor:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            torch.Tensor: Logits for each image and class
        """
        # Preprocess all images
        image_tensors = []
        for image in images:
            image_tensor = self.preprocess_image(image)
            image_tensors.append(image_tensor)
        
        # Stack into a batch
        batch_tensor = torch.cat(image_tensors, dim=0)
        
        # Get predictions
        logits = self.forward(batch_tensor)
        
        return logits
    
    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return self.class_names.copy()
    
    def save_text_embeddings(self, filepath: str):
        """Save the text embeddings to a file."""
        torch.save({
            'text_embeddings': self.text_embeddings,
            'class_names': self.class_names,
            'backbone': self.backbone,
            'use_templates': self.use_templates,
            'dataset_name': getattr(self, 'dataset_name', 'unknown')
        }, filepath)
        print(f"Text embeddings saved to {filepath}")
    
    def load_text_embeddings(self, filepath: str):
        """Load text embeddings from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.text_embeddings = checkpoint['text_embeddings'].to(self.device)
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        print(f"Text embeddings loaded from {filepath}")
        
MODEL_ZOO = {
    'r18': "resnet18",
    'r50': "resnet50",
    'r50_gn': "resnet50_gn",
    'r101': "resnet101",
    'r152': "resnet152",

    'fastvit-s12': 'fastvit_s12.apple_in1k',
    'mvitv2-1.0': 'mobilenetv2_100.ra_in1k',

    'vitt16': 'vit_tiny_patch16_224',
    'vits16': 'vit_small_patch16_224',
    'vitb16': 'vit_base_patch16_224',
    'vitb32': 'vit_base_patch32_224',
    'vitl16': 'vit_large_patch16_224',
    # 'vitl32': 'vit_large_patch32_224',
    'vith14': 'vit_huge_patch14_224',
    'vitg14': 'vit_giant_patch14_224',
    'vitgg14': 'vit_gigantic_patch14_224',
    'deitb16': 'deit_base_patch16_224',
    'deit3b16': 'deit3_base_patch16_224',
    
    # 'vitb32_clip': 'vit_base_patch32_clip_224',
    # 'vitb16_clip': 'vit_base_patch16_clip_224',
    # 'vitl14_clip': 'vit_large_patch14_clip_224',
    # 'vith14_clip': 'vit_huge_patch14_clip_224',
    # 'vitg14_clip': 'vit_giant_patch14_clip_224',
    # 'vitgg14_clip': 'vit_gigantic_patch14_clip_224',
}


CLIP_MODEL_ZOO ={
    'vitb32_clip': 'ViT-B/32',
    'vitb16_clip': 'ViT-B/16',
    'vitl14_clip': 'ViT-L/14',
}

def prepare_pretrained_model(model_name):
    # if 'vit' in model_name:
    #     model = timm.create_model(MODEL_ZOO[model_name], pretrained=True)
        
    # elif 'resnet' in model_name:
    #     # num_layers = re.findall(r'(\d+)', model_name)[0]
    #     # pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
    #     # model = models.__dict__[model_name](weights=pretrain_weights)
    if model_name in MODEL_ZOO:
        model = timm.create_model(MODEL_ZOO[model_name], pretrained=True)
    elif 'clip' in model_name:
        model = CLIPZeroShotClassifier(backbone=CLIP_MODEL_ZOO[model_name])
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    # if restore_weight is not None:
    #     state_dict = torch.load(restore_weight)['model']
    #     model.load_state_dict(state_dict)
    #     assert model_name in restore_weight, f'{model_name} is not in {restore_weight}'
    #     print(f'Restored model weight from {restore_weight}')
        
    return model

# def prepare_vlm_distilled_model(model_name, num_classes, restore_weight=None):
#     if 'vit' in model_name:
#         # model_dict  = {'b': 'base', 'l': 'large', 'h': 'huge'}
#         # patch_size = re.findall(r'(\d+)', model_name)[0]
#         # model_size = model_name[3]
#         # model = timm.create_model(f'vit_{model_dict[model_size]}_patch{patch_size}_224', pretrained=True)
#         model = timm.create_model(MODEL_ZOO[model_name], pretrained=True)
#         model.head = torch.nn.Linear(model.head.in_features, num_classes)
        
#     elif 'resnet' in model_name:
#         num_layers = re.findall(r'(\d+)', model_name)[0]
#         pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
#         model = models.__dict__[model_name](weights=pretrain_weights)
#         model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
#     else:
#         raise NotImplementedError(f'{model_name} is not supported')
    
#     if restore_weight is not None:
#         state_dict = torch.load(restore_weight)['model']
#         model.load_state_dict(state_dict)
#         assert model_name in restore_weight, f'{model_name} is not in {restore_weight}'
#         print(f'Restored model weight from {restore_weight}')
        
#     return model

# def prepare_student_model(model_name, mode='linear', lr=1e-3):
#     encoder_params, clf_params = [], []
#     if 'vit' in model_name:
#         # patch_size = re.findall(r'(\d+)', model_name)[0]
#         # model = timm.create_model(f'vit_base_patch{patch_size}_224', pretrained=True)
#         model = timm.create_model(MODEL_ZOO[model_name], pretrained=True)
#         model.head = torch.nn.Linear(model.head.in_features, model.head.out_features)
        
#         for name, param in model.named_parameters():
#             if 'head' not in name:
#                 encoder_params.append(param)
#             else:
#                 clf_params.append(param)
#     elif 'resnet' in model_name:
#         num_layers = re.findall(r'(\d+)', model_name)[0]
#         pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
#         model = models.__dict__[model_name](weights=pretrain_weights)
#         model.fc = torch.nn.Linear(model.fc.in_features, model.fc.out_features)
        
#         for name, param in model.named_parameters():
#             if 'fc' not in name:
#                 encoder_params.append(param)
#             else:
#                 clf_params.append(param)
        
#     else:
#         raise NotImplementedError(f'{model_name} is not supported')
    
#     if mode == 'linear':
#         for param in encoder_params:
#             param.requires_grad = False
#         for param in clf_params:
#             param.requires_grad = True    
#         optimizer = torch.optim.AdamW(clf_params, lr=lr)
        
#     elif mode == 'full':
#         for param in encoder_params:
#             param.requires_grad = True
#         for param in clf_params:
#             param.requires_grad = True
            
#         optimizer = torch.optim.AdamW(
#             [
#                 {'params': encoder_params, 'lr': lr/10},
#                 {'params': clf_params, 'lr': lr}
#             ]
#         )
#     else:
#         raise NotImplementedError(f'{mode} is not supported')
    
#     return model, optimizer

# def prepare_vlm_student_model(model_name, mode='linear', lr=1e-3, num_classes=1000):
#     encoder_params, clf_params = [], []
#     if 'vit' in model_name:
#         # patch_size = re.findall(r'(\d+)', model_name)[0]
#         # model = timm.create_model(f'vit_base_patch{patch_size}_224', pretrained=True)
#         model = timm.create_model(MODEL_ZOO[model_name], pretrained=True)
#         model.head = torch.nn.Linear(model.head.in_features, num_classes)
        
#         for name, param in model.named_parameters():
#             if 'head' not in name:
#                 encoder_params.append(param)
#             else:
#                 clf_params.append(param)
#     elif 'resnet' in model_name:
#         num_layers = re.findall(r'(\d+)', model_name)[0]
#         pretrain_weights = models.__dict__[f'ResNet{num_layers}_Weights'].IMAGENET1K_V1
#         model = models.__dict__[model_name](weights=pretrain_weights)
#         model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
#         for name, param in model.named_parameters():
#             if 'fc' not in name:
#                 encoder_params.append(param)
#             else:
#                 clf_params.append(param)
        
#     else:
#         raise NotImplementedError(f'{model_name} is not supported')
    
#     if mode == 'linear':
#         for param in encoder_params:
#             param.requires_grad = False
#         for param in clf_params:
#             param.requires_grad = True    
#         optimizer = torch.optim.AdamW(clf_params, lr=lr)
        
#     elif mode == 'full':
#         for param in encoder_params:
#             param.requires_grad = True
#         for param in clf_params:
#             param.requires_grad = True
            
#         optimizer = torch.optim.AdamW(
#             [
#                 {'params': encoder_params, 'lr': lr/10},
#                 {'params': clf_params, 'lr': lr}
#             ]
#         )
#     else:
#         raise NotImplementedError(f'{mode} is not supported')
    
#     return model, optimizer